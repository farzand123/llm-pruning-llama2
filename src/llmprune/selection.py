"""Selection rules shared by all pruning methods (pure NumPy, no PyTorch).

Everything here is deterministic: ties are always broken by flat index, so the same
scores always give the same pruning recipe.

Thesis mapping
--------------
* heads  (Chapter III.C.2): rank all L*H heads globally, remove floor(r * L * H) lowest.
  A pure global ranking can remove every head of a layer; the implementation keeps at
  least ``min_heads_per_layer`` (default 1) and *reports* whether that rule was binding
  (thesis Table 15).
* layers (Chapter III.C.3): remove floor(r * L) blocks with the lowest Block Influence.
* magnitude (Chapter III.C.1): one global threshold over all prunable weights. For 16-bit
  weights the threshold is found exactly from a histogram of magnitude bit patterns.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

_EPS = 1e-9  # guards floor() against products such as 0.3 * 10 = 2.9999999999999996


def n_to_remove(ratio: float, total: int) -> int:
    """floor(ratio * total), robust to floating-point under-rounding."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio must be in [0, 1), got {ratio!r}")
    return int(np.floor(ratio * total + _EPS))


# --------------------------------------------------------------------------- heads
def select_heads_to_remove(
    scores: Sequence[Sequence[float]],
    ratio: float,
    min_heads_per_layer: int = 1,
) -> Tuple[np.ndarray, Dict]:
    """Choose the attention heads to delete.

    Parameters
    ----------
    scores : array [n_layers, n_heads]; higher = more important.
    ratio : fraction of *all* heads to remove (thesis: 10..70 %).
    min_heads_per_layer : a layer is never reduced below this many heads (>= 1, because
        a layer without heads would need a special attention module).

    Returns
    -------
    remove_mask : bool array [n_layers, n_heads]; True = head is removed.
    info : dict with the realised counts and whether the per-layer minimum mattered.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError("scores must have shape [n_layers, n_heads]")
    if not np.isfinite(s).all():
        raise ValueError("head scores contain NaN or inf; check the calibration pass")
    if min_heads_per_layer < 1:
        raise ValueError("min_heads_per_layer must be >= 1")
    n_layers, n_heads = s.shape
    if min_heads_per_layer > n_heads:
        raise ValueError("min_heads_per_layer exceeds the number of heads")

    n_target = n_to_remove(ratio, n_layers * n_heads)
    capacity = n_layers * (n_heads - min_heads_per_layer)
    if n_target > capacity:
        raise ValueError(
            f"cannot remove {n_target} heads while keeping {min_heads_per_layer} per layer "
            f"(capacity {capacity})"
        )

    order = np.argsort(s.ravel(), kind="stable")  # ascending importance, ties by index

    # What an unconstrained global ranking would do (reported only).
    pure = np.zeros(n_layers * n_heads, dtype=bool)
    pure[order[:n_target]] = True
    pure = pure.reshape(n_layers, n_heads)
    emptied = [int(l) for l in range(n_layers) if pure[l].all()]

    remove = np.zeros(n_layers * n_heads, dtype=bool)
    kept = [n_heads] * n_layers
    done = skipped = 0
    binding = set()
    for flat in order:
        if done == n_target:
            break
        layer = int(flat) // n_heads
        if kept[layer] <= min_heads_per_layer:
            skipped += 1
            binding.add(layer)
            continue
        remove[flat] = True
        kept[layer] -= 1
        done += 1
    if done != n_target:  # pragma: no cover - excluded by the capacity check above
        raise RuntimeError("internal error: could not remove the requested number of heads")

    info = {
        "n_heads_total": int(n_layers * n_heads),
        "n_remove_target": int(n_target),
        "n_removed": int(done),
        "n_kept": int(n_layers * n_heads - done),
        "min_heads_per_layer": int(min_heads_per_layer),
        "layers_emptied_by_pure_global_ranking": emptied,
        "layers_where_min_rule_was_binding": sorted(int(l) for l in binding),
        "candidates_skipped_by_min_rule": int(skipped),
        "heads_kept_per_layer": [int(k) for k in kept],
    }
    return remove.reshape(n_layers, n_heads), info


def keep_lists(remove_mask: np.ndarray) -> List[List[int]]:
    """Indices of the heads that are *kept* in each layer."""
    m = np.asarray(remove_mask, dtype=bool)
    return [[int(h) for h in np.flatnonzero(~row)] for row in m]


def head_channel_indices(keep: Sequence[int], head_dim: int) -> np.ndarray:
    """Channel indices (rows of W_Q/W_K/W_V, columns of W_O) of the kept heads.

    Head h owns channels [h * head_dim, (h + 1) * head_dim).
    """
    if len(keep) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(h * head_dim, (h + 1) * head_dim, dtype=np.int64) for h in keep])


# --------------------------------------------------------------------------- layers
def select_layers_to_remove(bi_scores: Sequence[float], ratio: float) -> List[int]:
    """Indices (ascending) of the floor(ratio * L) blocks with the lowest Block Influence."""
    bi = np.asarray(bi_scores, dtype=np.float64)
    if bi.ndim != 1:
        raise ValueError("bi_scores must be one-dimensional")
    if not np.isfinite(bi).all():
        raise ValueError("Block Influence scores contain NaN or inf")
    n_remove = n_to_remove(ratio, len(bi))
    if n_remove >= len(bi):
        raise ValueError("would remove every layer")
    order = np.argsort(bi, kind="stable")
    return sorted(int(i) for i in order[:n_remove])


# --------------------------------------------------------------------------- magnitude
def threshold_from_histogram(hist: Sequence[int], target: int) -> Tuple[int, int, float]:
    """Exact global magnitude threshold from a histogram of magnitude bit patterns.

    For bfloat16 / float16 the 15 low bits of the raw 16-bit word grow monotonically with
    |w|, so ``hist[p]`` = number of weights whose magnitude pattern equals ``p``
    (0 .. 32767). A weight is pruned iff its pattern is ``< p_star``.

    The returned ``p_star`` is the pattern for which the number of pruned weights is
    closest to ``target`` (ties in |w| make an exact count impossible in general).

    Returns (p_star, pruned_count, realised_ratio).
    """
    h = np.asarray(hist, dtype=np.int64)
    total = int(h.sum())
    if not 0 <= target <= total:
        raise ValueError("target outside [0, total]")
    cum_lt = np.concatenate([[0], np.cumsum(h)])  # cum_lt[p] = #weights with pattern < p
    p_star = int(np.argmin(np.abs(cum_lt - target)))
    pruned = int(cum_lt[p_star])
    return p_star, pruned, (pruned / total if total else 0.0)
