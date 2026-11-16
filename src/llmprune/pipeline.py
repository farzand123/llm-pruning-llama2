"""Running the experiment matrix (PyTorch): recipe -> LoRA -> evaluation -> efficiency, per condition.

Every stage writes its result file atomically and only when it is complete, so an interrupted run
restarts where it stopped (see ``status.plan_actions``). Models are loaded once per stage and freed
afterwards; the efficiency benchmark loads a fresh copy for every device setting.
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .config import (Condition, adapter_dir, efficiency_path, eval_path, recipe_path)
from .io_utils import (environment_info, file_lock, git_commit, read_json, utc_now, write_json)
from .modeling import free_memory
from .status import STAGES, plan_actions, recipe_sha

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- helpers
def load_base(cfg: Dict[str, Any], model_key: str, device_map):
    from .modeling import load_model

    spec = cfg["models"][model_key]
    return load_model(spec["name_or_path"], cfg["runtime"]["dtype"], device_map, spec.get("revision"),
                      cfg["runtime"].get("attn_implementation"))


def get_calibration(cfg: Dict[str, Any], tokenizer) -> torch.Tensor:
    from .data import get_c4_calibration

    cal = cfg["calibration"]
    return get_c4_calibration(tokenizer, int(cal["n_samples"]), int(cal["seqlen"]), int(cfg["seed"]),
                              cal.get("c4_shard", "en/c4-train.00000-of-01024.json.gz"), cal.get("c4_local_path"),
                              cache_dir=cfg["paths"].get("cache", "cache"))


def provenance(cfg: Dict[str, Any], cond: Condition, **extra) -> Dict[str, Any]:
    from .modeling import hardware_info

    return {"condition": cond.id, "no": cond.no, "recipe_sha256": recipe_sha(cfg, cond), "git": git_commit(str(REPO_ROOT)),
            "timestamp": utc_now(), "config": cfg.get("_config_path"), "seed": cfg["seed"],
            "environment": environment_info(), "hardware": hardware_info(), **extra}


def load_recipe(cfg: Dict[str, Any], cond: Condition) -> Optional[Dict[str, Any]]:
    p = recipe_path(cfg, cond)
    return read_json(p) if p is not None and p.is_file() else None


def load_condition_model(cfg: Dict[str, Any], cond: Condition, device_map=None, with_adapter: bool = True):
    """Load the base model, replay the recipe and (optionally) attach the trained LoRA adapter.

    Use this to work with a pruned model in your own code:

        cfg = load_config("configs/full.yaml")
        cond = next(c for c in expand_conditions(cfg) if c.id == "7b-heads-30-lora")
        model, tok = load_condition_model(cfg, cond)
    """
    from .lora import load_adapter
    from .pruning import apply_recipe

    model, tok = load_base(cfg, cond.model_key, cfg["runtime"]["device_map"] if device_map is None else device_map)
    if cond.recipe_key:
        recipe = load_recipe(cfg, cond)
        if recipe is None:
            raise FileNotFoundError(f"recipe {recipe_path(cfg, cond)} does not exist; run `llmprune run --only {cond.id}`")
        apply_recipe(model, recipe)
    if with_adapter and cond.lora_rank:
        model = load_adapter(model, adapter_dir(cfg, cond))
    return model, tok


# --------------------------------------------------------------------------- stages
def stage_recipe(cfg: Dict[str, Any], cond: Condition, force: bool = False) -> None:
    """Compute and store the pruning recipe. Safe against two shards needing the same recipe."""
    from .pruning import make_recipe

    path = recipe_path(cfg, cond)
    started = time.time()
    with file_lock(path.with_suffix(".lock")):
        if path.is_file() and (not force or path.stat().st_mtime >= started):
            log.info("[%s] recipe %s already exists", cond.id, path.name)
            return
        log.info("[%s] computing recipe %s", cond.id, path.name)
        model, tok = load_base(cfg, cond.model_key, cfg["runtime"]["device_map"])
        calib = get_calibration(cfg, tok) if cond.method in ("heads", "layers") else None
        recipe = make_recipe(model, calib, cfg, cond, cfg["models"][cond.model_key])
        write_json(path, recipe)
        log.info("[%s] recipe written (%.0f s)", cond.id, recipe["seconds"])


def stage_lora(cfg: Dict[str, Any], cond: Condition) -> None:
    from .lora import train_lora
    from .pruning import apply_recipe

    log.info("[%s] LoRA recovery, rank %d", cond.id, cond.lora_rank)
    model, tok = load_base(cfg, cond.model_key, cfg["lora"].get("device_map", "single"))
    if cond.recipe_key:
        apply_recipe(model, load_recipe(cfg, cond))
    summary = train_lora(cfg, cond, model, tok, adapter_dir(cfg, cond), provenance(cfg, cond))
    log.info("[%s] adapter saved: %d steps, final loss %.4f, base weights unchanged: %s", cond.id, summary["steps"],
             summary["final_loss"], summary["base_weights_unchanged"])


def stage_eval(cfg: Dict[str, Any], cond: Condition, kinds: Sequence[str]) -> None:
    from .evaluation import run_eval

    if not kinds:
        return
    model, tok = load_condition_model(cfg, cond)
    extra: Dict[str, Any] = {}
    if cond.lora_rank:
        ts = adapter_dir(cfg, cond) / "train_summary.json"
        extra["adapter_finished"] = read_json(ts).get("finished") if ts.is_file() else None
    for kind in kinds:
        t0 = time.time()
        log.info("[%s] evaluating %s", cond.id, kind)
        result = run_eval(kind, model, tok, cfg)
        result["provenance"] = provenance(cfg, cond, seconds=round(time.time() - t0, 1), **extra)
        write_json(eval_path(cfg, cond, kind), result)
        headline = {"ppl": "wikitext2_ppl", "mmlu": "macro_average_pct", "glue": "average_pct"}[kind]
        log.info("[%s] %s = %.3f", cond.id, kind, result[headline])


def available_settings(settings: Sequence[str]) -> List[str]:
    """'auto' is only meaningful with at least two visible GPUs (with one it is identical to 'single')."""
    n = torch.cuda.device_count()
    keep = []
    for s in settings:
        if s == "auto" and n < 2:
            log.warning("skipping efficiency setting 'auto': only %d GPU visible (it would equal 'single')", n)
            continue
        keep.append(s)
    return keep


def stage_efficiency(cfg: Dict[str, Any], cond: Condition, settings: Sequence[str]) -> None:
    from .benchmark import allocated_bytes, benchmark_generation

    ec = cfg["efficiency"]
    extra: Dict[str, Any] = {}
    if cond.lora_rank:
        ts = adapter_dir(cfg, cond) / "train_summary.json"
        extra["adapter_finished"] = read_json(ts).get("finished") if ts.is_file() else None
    for setting in available_settings(settings):
        free_memory()
        baseline = allocated_bytes()
        log.info("[%s] efficiency benchmark, device setting %r", cond.id, setting)
        model, tok = load_condition_model(cfg, cond, device_map=setting)
        t0 = time.time()
        res = benchmark_generation(model, tok, prompt_len=int(ec["prompt_len"]), new_tokens=int(ec["new_tokens"]),
                                   batch=int(ec["batch"]), runs=int(ec["runs"]), warmup=int(ec["warmup"]),
                                   seed=int(cfg["seed"]), baseline_allocated=baseline)
        res["setting"] = setting
        res["adapter"] = "unmerged" if cond.lora_rank else None
        res["provenance"] = provenance(cfg, cond, seconds=round(time.time() - t0, 1), **extra)
        write_json(efficiency_path(cfg, cond, setting), res)
        log.info("[%s] %s: %.1f tok/s, weights %s GB, peak %s GB", cond.id, setting, res["tokens_per_s_mean"],
                 res["weights_allocated_gb"], res["peak_allocated_gb"])
        del model, tok  # drop the references before measuring the next setting
        free_memory()


# --------------------------------------------------------------------------- orchestration
def run_condition(cfg: Dict[str, Any], cond: Condition, stages: Sequence[str] = STAGES, force: bool = False) -> Dict[str, Any]:
    """Run whatever is missing (or stale) for one condition. Returns the executed plan."""
    plan = plan_actions(cfg, cond, stages, force)
    run = plan["run"]
    if not any(run.values()):
        log.info("[%s] complete", cond.id)
        return plan
    try:
        if run["recipe"]:
            stage_recipe(cfg, cond, force=force and "recipe" in stages)
            free_memory()
        if run["lora"]:
            stage_lora(cfg, cond)
            free_memory()
        if run["eval"]:
            stage_eval(cfg, cond, plan["eval_kinds"])
            free_memory()
        if run["efficiency"]:
            stage_efficiency(cfg, cond, plan["efficiency_settings"])
    finally:
        free_memory()
    return plan


def run_matrix(cfg: Dict[str, Any], conds: Sequence[Condition], stages: Sequence[str] = STAGES, force: bool = False,
               continue_on_error: bool = False) -> List[str]:
    """Run all conditions in order; returns the ids of the conditions that failed (empty on success)."""
    failed: List[str] = []
    for i, cond in enumerate(conds, 1):
        log.info("=== %d/%d  %s  (Table 17 No. %d) ===", i, len(conds), cond.id, cond.no)
        try:
            run_condition(cfg, cond, stages, force)
        except Exception:
            if not continue_on_error:
                raise
            log.error("[%s] FAILED\n%s", cond.id, traceback.format_exc())
            failed.append(cond.id)
    return failed
