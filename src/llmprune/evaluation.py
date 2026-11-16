"""Evaluation (PyTorch): WikiText-2 perplexity and lm-evaluation-harness tasks (MMLU, GLUE).

Every function returns a plain dict that ``pipeline`` writes to ``results/<condition>/eval_<kind>.json``;
the keys are the ones ``report.py`` reads.
"""
from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from . import metrics as M
from .data import get_wikitext2_test_ids
from .io_utils import package_versions
from .modeling import embed_device

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- perplexity
@torch.no_grad()
def sliding_window_nll(model, ids: torch.Tensor, window: int, stride: int,
                       max_windows: Optional[int] = None) -> Dict[str, Any]:
    """Exact token-level negative log-likelihood with overlapping windows.

    Window k covers tokens [k*stride, k*stride + window). Only tokens that no earlier window has scored
    are scored (so each token has up to ``window - 1`` tokens of context) and the first token of the text
    is never scored. The sum is over tokens, not an approximation by window length as in some references.
    """
    if not 0 < stride <= window:
        raise ValueError("need 0 < stride <= window")
    dev = embed_device(model)
    n = ids.size(1)
    nll_sum, n_tok, prev_end, n_windows = 0.0, 0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        chunk = ids[:, begin:end].to(dev)
        t = chunk.size(1)
        first_new = max(prev_end - begin, 1)  # window-relative index of the first token not yet scored
        if t - first_new <= 0:
            break
        logits = model(chunk, use_cache=False).logits
        pred = logits[0, first_new - 1:t - 1].float()
        target = chunk[0, first_new:t].to(pred.device)
        nll_sum += float(F.cross_entropy(pred, target, reduction="sum"))
        n_tok += t - first_new
        prev_end = end
        n_windows += 1
        if end == n or (max_windows is not None and n_windows >= max_windows):
            break
    return {"nll_sum": nll_sum, "n_tokens": n_tok, "n_windows": n_windows,
            "nll_per_token": nll_sum / max(n_tok, 1), "ppl": math.exp(nll_sum / max(n_tok, 1))}


def evaluate_ppl(model, tokenizer, ev_cfg: Dict[str, Any]) -> Dict[str, Any]:
    window, stride = int(ev_cfg["ppl"]["window"]), int(ev_cfg["ppl"]["stride"])
    ids = get_wikitext2_test_ids(tokenizer)
    limit = ev_cfg.get("limit")
    r = sliding_window_nll(model, ids, window, stride, max_windows=int(limit) if limit else None)
    return {
        "wikitext2_ppl": r["ppl"], "nll_per_token": r["nll_per_token"], "n_tokens": r["n_tokens"],
        "n_windows": r["n_windows"], "text_tokens": int(ids.size(1)),
        "protocol": {"split": "test", "window": window, "stride": stride, "limit_windows": limit,
                     "text": "'\\n\\n'.join(wikitext-2-raw-v1 test)"},
    }


# --------------------------------------------------------------------------- lm-evaluation-harness
def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keyword arguments that ``fn`` names explicitly (lm_eval's signature differs between releases)."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def run_harness(model, tokenizer, tasks: Sequence[str], num_fewshot: int, batch_size: int,
                limit: Optional[int], seed: int, add_bos_token: bool = False) -> Dict[str, Any]:
    """Run ``lm_eval.simple_evaluate`` on an already loaded (possibly PEFT-wrapped, possibly pruned) model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm_kwargs = {"pretrained": model, "tokenizer": tokenizer, "batch_size": batch_size, "add_bos_token": add_bos_token}
    accepted = _filter_kwargs(HFLM.__init__, lm_kwargs)
    if "add_bos_token" not in accepted and add_bos_token:
        log.warning("this lm_eval version has no add_bos_token option; evaluating without BOS")
    lm = HFLM(**accepted)

    kwargs = {
        "model": lm, "tasks": list(tasks), "num_fewshot": num_fewshot, "limit": limit,
        "log_samples": False, "bootstrap_iters": 0, "verbosity": "WARNING",
        "random_seed": seed, "numpy_random_seed": seed, "torch_random_seed": seed,
    }
    out = lm_eval.simple_evaluate(**_filter_kwargs(lm_eval.simple_evaluate, kwargs))
    return {"results": out["results"], "groups": out.get("groups") or {}, "versions": out.get("versions"),
            "n_shot": out.get("n-shot")}


def _protocol(ev_cfg: Dict[str, Any], tasks: Sequence[str], num_fewshot: int, seed: int) -> Dict[str, Any]:
    return {"tasks": list(tasks), "num_fewshot": num_fewshot, "limit": ev_cfg.get("limit"),
            "batch_size": ev_cfg.get("batch_size"), "add_bos_token": bool(ev_cfg.get("add_bos_token", False)),
            "seed": seed, "lm_eval": package_versions(["lm_eval"]).get("lm_eval")}


def evaluate_mmlu(model, tokenizer, ev_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    nfs = int(ev_cfg["mmlu"]["num_fewshot"])
    raw = run_harness(model, tokenizer, ["mmlu"], nfs, int(ev_cfg.get("batch_size", 8)), ev_cfg.get("limit"),
                      seed, bool(ev_cfg.get("add_bos_token", False)))
    out = M.summarise_mmlu(raw["results"], raw["groups"])
    out["protocol"] = _protocol(ev_cfg, ["mmlu"], nfs, seed)
    return out


def evaluate_glue(model, tokenizer, ev_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    tasks: List[str] = list(ev_cfg["glue"]["tasks"])
    nfs = int(ev_cfg["glue"]["num_fewshot"])
    raw = run_harness(model, tokenizer, tasks, nfs, int(ev_cfg.get("batch_size", 8)), ev_cfg.get("limit"),
                      seed, bool(ev_cfg.get("add_bos_token", False)))
    out = M.summarise_glue(raw["results"], tasks)
    out["protocol"] = _protocol(ev_cfg, tasks, nfs, seed)
    return out


def run_eval(kind: str, model, tokenizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch on the evaluation kind ('ppl', 'mmlu' or 'glue')."""
    ev, seed = cfg["evaluation"], int(cfg["seed"])
    if kind == "ppl":
        return evaluate_ppl(model, tokenizer, ev)
    if kind == "mmlu":
        return evaluate_mmlu(model, tokenizer, ev, seed)
    if kind == "glue":
        return evaluate_glue(model, tokenizer, ev, seed)
    raise ValueError(f"unknown evaluation kind {kind!r}")
