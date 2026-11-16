import numpy as np
import pytest

from llmprune import selection as sel


def test_head_counts_match_thesis_ratios():
    s = np.random.default_rng(0).random((32, 32))
    expected = {0.1: 102, 0.2: 204, 0.3: 307, 0.4: 409, 0.5: 512, 0.7: 716}
    for ratio, n in expected.items():
        mask, info = sel.select_heads_to_remove(s, ratio)
        assert mask.sum() == n == info["n_removed"]
        assert info["n_kept"] == 1024 - n


def test_layer_counts_match_thesis_ratios():
    bi = np.random.default_rng(1).random(32)
    assert [len(sel.select_layers_to_remove(bi, r)) for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)] == [3, 6, 9, 12, 16, 22]


def test_floor_is_robust_to_float_error():
    assert sel.n_to_remove(0.3, 10) == 3
    assert sel.n_to_remove(0.7, 10) == 7
    assert sel.n_to_remove(0.6, 5) == 3


def test_heads_removed_are_the_lowest_scores():
    s = np.arange(12, dtype=float).reshape(3, 4)  # increasing importance
    mask, _ = sel.select_heads_to_remove(s, 0.25, min_heads_per_layer=1)
    assert mask.sum() == 3
    assert set(map(tuple, np.argwhere(mask))) == {(0, 0), (0, 1), (0, 2)}


def test_min_heads_rule_and_reporting():
    s = np.random.default_rng(2).random((8, 8))
    s[0] = -1.0  # layer 0 is the least important layer by far
    mask, info = sel.select_heads_to_remove(s, 0.3, min_heads_per_layer=1)
    assert info["layers_emptied_by_pure_global_ranking"] == [0]
    assert info["layers_where_min_rule_was_binding"] == [0]
    assert info["heads_kept_per_layer"][0] == 1
    assert (~mask).sum(axis=1).min() >= 1
    assert mask.sum() == info["n_remove_target"]


def test_selection_is_deterministic_with_ties():
    s = np.ones((4, 4))
    a, _ = sel.select_heads_to_remove(s, 0.5)
    b, _ = sel.select_heads_to_remove(s.copy(), 0.5)
    assert (a == b).all()


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sel.select_heads_to_remove(np.array([[1.0, np.nan]]), 0.1)
    with pytest.raises(ValueError):
        sel.select_heads_to_remove(np.ones((2, 2)), 0.9, min_heads_per_layer=2)
    with pytest.raises(ValueError):
        sel.select_layers_to_remove([1.0, float("inf")], 0.1)
    with pytest.raises(ValueError):
        sel.n_to_remove(1.0, 10)


def test_keep_lists_and_channel_indices():
    mask = np.array([[True, False, False], [False, False, True]])
    keep = sel.keep_lists(mask)
    assert keep == [[1, 2], [0, 1]]
    idx = sel.head_channel_indices(keep[0], head_dim=4)
    assert idx.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert sel.head_channel_indices([], 4).size == 0


def bf16_pattern(x):
    """Magnitude bit pattern of the bfloat16 obtained by truncating float32."""
    return ((np.asarray(x, dtype=np.float32).view(np.uint32) >> 16) & 0x7FFF).astype(np.int64)


def test_bf16_pattern_is_monotone_in_magnitude():
    x = np.sort(np.abs(np.random.default_rng(3).normal(size=5000)).astype(np.float32))
    p = bf16_pattern(x)
    assert (np.diff(p) >= 0).all()
    assert (bf16_pattern(-x) == p).all()  # sign bit is masked away


def test_histogram_threshold_hits_target_closely():
    w = np.random.default_rng(4).normal(scale=0.02, size=200_000).astype(np.float32)
    pat = bf16_pattern(w)
    hist = np.bincount(pat, minlength=32768)
    for ratio in (0.1, 0.3, 0.5, 0.7):
        target = int(ratio * w.size)
        p_star, pruned, realised = sel.threshold_from_histogram(hist, target)
        assert pruned == int((pat < p_star).sum())
        # bf16 has few distinct magnitudes, so allow the size of one histogram bin
        assert abs(pruned - target) <= hist.max()
        assert abs(realised - ratio) < 0.01


def test_histogram_threshold_edges():
    hist = np.array([5, 0, 3, 2])
    assert sel.threshold_from_histogram(hist, 0)[:2] == (0, 0)
    p, pruned, ratio = sel.threshold_from_histogram(hist, 10)
    assert pruned == 10 and ratio == 1.0
    with pytest.raises(ValueError):
        sel.threshold_from_histogram(hist, 11)
