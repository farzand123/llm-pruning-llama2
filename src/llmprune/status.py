"""Which outputs of the matrix exist, which are stale, and what a run would do (pure Python).

Stages of one condition, in dependency order:

    recipe  ->  lora  ->  eval        (a LoRA-recovered condition evaluates the trained adapter)
                      ->  efficiency

Every result file stores ``provenance.recipe_sha256`` (and, for adapters, ``finished``). If a recipe is
recomputed, everything built on the old one is reported as STALE and is recomputed by the next ``run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import (Condition, adapter_dir, efficiency_path, eval_path, recipe_path)
from .io_utils import read_json, sha256_file

STAGES = ("recipe", "lora", "eval", "efficiency")
DOWNSTREAM = {"recipe": ("lora", "eval", "efficiency"), "lora": ("eval", "efficiency"), "eval": (), "efficiency": ()}


def output_map(cfg: Dict[str, Any], cond: Condition, settings: Optional[Sequence[str]] = None) -> Dict[str, List[Path]]:
    """stage -> files whose existence means the stage is finished."""
    out: Dict[str, List[Path]] = {s: [] for s in STAGES}
    if cond.is_noop:
        return out
    if cond.recipe_key:
        out["recipe"].append(recipe_path(cfg, cond))
    if cond.lora_rank:
        out["lora"].append(adapter_dir(cfg, cond) / "train_summary.json")
    out["eval"] = [eval_path(cfg, cond, k) for k in cond.evals]
    if cond.efficiency:
        out["efficiency"] = [efficiency_path(cfg, cond, s) for s in (settings or cfg["efficiency"]["settings"])]
    return out


def recipe_sha(cfg: Dict[str, Any], cond: Condition) -> Optional[str]:
    p = recipe_path(cfg, cond)
    return sha256_file(p) if p is not None and p.is_file() else None


def _is_stale(path: Path, sha: Optional[str], adapter_finished: Optional[str]) -> bool:
    try:
        prov = (read_json(path) or {}).get("provenance") or {}
    except Exception:  # unreadable / truncated file
        return True
    if sha is not None and prov.get("recipe_sha256") not in (None, sha):
        return True
    if adapter_finished is not None and prov.get("adapter_finished") not in (None, adapter_finished):
        return True
    return False


def condition_status(cfg: Dict[str, Any], cond: Condition) -> Dict[str, Dict[str, List[Path]]]:
    """stage -> {"present": [...], "missing": [...], "stale": [...]} (paths)."""
    outs = output_map(cfg, cond)
    sha = recipe_sha(cfg, cond)
    finished = None
    if cond.lora_rank:
        ts = adapter_dir(cfg, cond) / "train_summary.json"
        if ts.is_file():
            try:
                finished = (read_json(ts) or {}).get("finished")
            except Exception:
                finished = None
    res: Dict[str, Dict[str, List[Path]]] = {}
    for stage, paths in outs.items():
        present = [p for p in paths if p.is_file()]
        missing = [p for p in paths if not p.is_file()]
        stale: List[Path] = []
        if stage != "recipe":
            for p in present:
                if _is_stale(p, sha, finished if stage in ("eval", "efficiency") else None):
                    stale.append(p)
        res[stage] = {"present": present, "missing": missing, "stale": stale}
    return res


def is_complete(cfg: Dict[str, Any], cond: Condition) -> bool:
    st = condition_status(cfg, cond)
    return all(not v["missing"] and not v["stale"] for v in st.values())


def plan_actions(cfg: Dict[str, Any], cond: Condition, stages: Sequence[str] = STAGES, force: bool = False) -> Dict[str, Any]:
    """What ``run`` would do for one condition.

    Returns {"run": {stage: bool}, "eval_kinds": [...], "efficiency_settings": [...]}.
    A stage that is not selected still runs if a selected stage needs its output (e.g. the recipe).
    """
    st = condition_status(cfg, cond)
    wanted = [s for s in STAGES if s in stages]
    forced = set()
    if force:
        for s in wanted:
            forced.add(s)
            forced.update(DOWNSTREAM[s])

    def path_todo(stage: str) -> List[Path]:
        if stage in forced:
            return st[stage]["present"] + st[stage]["missing"]
        return st[stage]["missing"] + st[stage]["stale"]

    work = {s: (s in wanted) and bool(path_todo(s)) for s in STAGES}
    run: Dict[str, bool] = {}
    for s in STAGES:
        own_incomplete = bool(st[s]["missing"] or st[s]["stale"])
        needed_later = any(work[d] for d in DOWNSTREAM[s])
        run[s] = bool(work[s] or (own_incomplete and needed_later))

    kinds = [k for k, p in zip(cond.evals, output_map(cfg, cond)["eval"]) if p in path_todo("eval")] if run["eval"] else []
    settings = []
    if run["efficiency"]:
        for s, p in zip(cfg["efficiency"]["settings"], output_map(cfg, cond)["efficiency"]):
            if p in path_todo("efficiency"):
                settings.append(s)
    return {"run": run, "eval_kinds": kinds, "efficiency_settings": settings}
