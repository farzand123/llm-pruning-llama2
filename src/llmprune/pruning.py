"""Creating and applying pruning *recipes* (PyTorch).

A recipe is a small JSON document that, together with the base model, reproduces a pruned model
exactly:

* heads     - ``heads_keep``: the retained head indices of every layer,
* layers    - ``layers_removed``: indices of the deleted blocks,
* magnitude - ``magnitude.pattern_threshold``: weights with a smaller magnitude bit pattern are zeroed.

``make_recipe`` computes the scores, edits the given model *in place* and returns the recipe;
``apply_recipe`` replays a stored recipe on a freshly loaded base model.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import counting as C
from . import selection as sel
from .data import calibration_hash
from .importance import block_influence, taylor_head_importance
from .io_utils import environment_info, git_commit, utc_now
from .modeling import (DTYPES, apply_head_pruning, apply_pattern_threshold, compare_logits, count_parameters,
                       describe_shapes, embed_device, logits_with_heads_masked, magnitude_prune, remove_layers,
                       unwrap)

log = logging.getLogger(__name__)
RECIPE_SCHEMA = 1

# Head pruning is verified against "same model with the removed heads masked out". In float32 the two
# agree to rounding error. In bfloat16 / float16 different matrix shapes pick different kernels, so only
# gross errors (wrong indices, misaligned W_O columns: relative error of order 1) are detectable, and
# only the relative logit error is a hard criterion: the argmax agreement of a heavily pruned model (whose
# logits are nearly flat) is merely reported, because rounding noise alone can flip many near-ties.
_TOLERANCE = {"float32": (1e-4, 0.999), "bfloat16": (0.2, 0.8), "float16": (0.2, 0.8)}


def check_head_verification(metrics: Dict[str, float], dtype: str) -> None:
    rel_max, agree_min = _TOLERANCE[dtype]
    hint = "Run `llmprune selftest`; set pruning.heads.verify: false to skip this check."
    if metrics["rel_l2"] > rel_max:
        raise RuntimeError(f"head pruning does not reproduce the masked reference ({metrics}; "
                           f"allowed rel_l2 <= {rel_max}). {hint}")
    if metrics["argmax_agreement"] < agree_min:
        if dtype == "float32":
            raise RuntimeError(f"head pruning does not reproduce the masked reference ({metrics}; "
                               f"argmax agreement must be >= {agree_min}). {hint}")
        log.warning("head pruning: argmax agreement with the masked reference is %.3f (< %.2f) although the relative logit "
                    "error is small (%.4f); expected for a heavily pruned model in %s", metrics["argmax_agreement"],
                    agree_min, metrics["rel_l2"], dtype)


def calibration_meta(calib: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cal = cfg["calibration"]
    return {"dataset": cal.get("dataset", "c4"), "shard": cal.get("c4_shard"), "n_samples": int(calib.size(0)),
            "seqlen": int(calib.size(1)), "seed": cfg["seed"], "token_sha256": calibration_hash(calib)}


def make_recipe(model, calib: Optional[torch.Tensor], cfg: Dict[str, Any], cond, model_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Score, prune ``model`` in place and describe what was done. ``calib`` is ignored for magnitude pruning."""
    base = unwrap(model)
    method, ratio = cond.method, cond.ratio
    pcfg = cfg["pruning"]
    dtype = cfg["runtime"]["dtype"]
    t0 = time.time()
    recipe: Dict[str, Any] = {
        "schema": RECIPE_SCHEMA, "id": cond.recipe_key, "model_key": cond.model_key,
        "model": {"name_or_path": model_spec["name_or_path"], "revision": model_spec.get("revision")},
        "method": method, "pct": cond.pct, "ratio": ratio, "dtype": dtype, "seed": cfg["seed"],
        "arch": asdict(C.arch_from_config(base.config)),
        "params_before": count_parameters(base), "shapes_before": describe_shapes(base),
    }

    if method == "heads":
        if calib is None:
            raise ValueError("head pruning needs a calibration sample")
        hcfg = pcfg["heads"]
        tb = int(hcfg.get("taylor_batch_size", 1))
        scores = taylor_head_importance(model, calib, batch_size=tb)
        remove_mask, info = sel.select_heads_to_remove(scores, ratio, int(hcfg.get("min_heads_per_layer", 1)))
        keep = sel.keep_lists(remove_mask)
        verify = bool(hcfg.get("verify", True))
        probe = calib[:1, :min(128, calib.size(1))]
        ref = logits_with_heads_masked(model, remove_mask, probe) if verify else None
        apply_head_pruning(model, keep)
        verification = None
        if verify:
            with torch.no_grad():
                cand = model(probe.to(embed_device(model)), use_cache=False).logits
            verification = compare_logits(ref, cand)
            check_head_verification(verification, dtype)
        recipe.update(
            calibration=calibration_meta(calib, cfg), heads_keep=keep, head_selection=info,
            head_scores=np.asarray(scores).tolist(),
            taylor={"criterion": "|sum f * dL/df| accumulated per batch (thesis Algorithm 1)",
                    "batch_size": tb, "n_sequences": int(calib.size(0))},
            verification=verification,
        )
    elif method == "layers":
        if calib is None:
            raise ValueError("layer pruning needs a calibration sample")
        variant = pcfg["layers"]["bi_variant"]
        bi = block_influence(model, calib, batch_size=int(cfg["calibration"].get("batch_size", 4)))
        removed = sel.select_layers_to_remove(bi[variant], ratio)
        remove_layers(model, removed)
        recipe.update(calibration=calibration_meta(calib, cfg), bi_variant=variant, bi_scores=bi, layers_removed=removed)
    elif method == "magnitude":
        recipe["magnitude"] = magnitude_prune(model, ratio)
    else:
        raise ValueError(f"unknown pruning method {method!r}")

    recipe["params_after"] = count_parameters(base)
    recipe["shapes_after"] = describe_shapes(base)
    recipe["created"] = utc_now()
    recipe["seconds"] = round(time.time() - t0, 1)
    recipe["git"] = git_commit()
    recipe["environment"] = environment_info()
    return recipe


def apply_recipe(model, recipe: Dict[str, Any]):
    """Replay a stored recipe on a freshly loaded, unpruned base model (in place)."""
    base = unwrap(model)
    if asdict(C.arch_from_config(base.config)) != recipe["arch"]:
        raise ValueError("recipe was made for a different architecture than the loaded model "
                         f"({recipe['arch']} vs {asdict(C.arch_from_config(base.config))})")
    have = next(base.parameters()).dtype
    if have != DTYPES[recipe["dtype"]]:
        raise ValueError(f"recipe was made with dtype {recipe['dtype']}, the model is loaded as {have}")
    method = recipe["method"]
    if method == "heads":
        apply_head_pruning(base, recipe["heads_keep"])
    elif method == "layers":
        remove_layers(base, recipe["layers_removed"])
    elif method == "magnitude":
        n = apply_pattern_threshold(base, recipe["magnitude"]["pattern_threshold"])
        if n != recipe["magnitude"]["zeroed"]:
            raise ValueError(f"recipe zeroes {recipe['magnitude']['zeroed']} weights but {n} fell below the stored threshold: "
                             "the base weights differ from the ones the recipe was made with (other revision or dtype?)")
    else:
        raise ValueError(f"unknown pruning method {method!r}")
    return model
