"""YAML configuration, experiment-matrix expansion and path conventions (pure Python).

The matrix reproduces thesis Table 17. A "condition" is one row of work: prune (or not),
optionally recover with LoRA, then evaluate and/or measure efficiency. Pruned models are
stored as small *recipes* (JSON), not as 13 GB checkpoints: none of the three methods
changes a surviving weight (heads are sliced, layers deleted, magnitude zeroes weights
below a stored threshold), so recipe + base model reproduce the pruned model exactly.
"""
from __future__ import annotations

import copy
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

METHODS = ("magnitude", "heads", "layers")
EVAL_KINDS = ("ppl", "glue", "mmlu")
_REQUIRED = (
    "seed", "paths", "models", "runtime", "calibration", "pruning",
    "evaluation", "efficiency", "lora", "matrix",
)


# --------------------------------------------------------------------------- loading
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> Dict[str, Any]:
    """Load a YAML config. A top-level ``extends: other.yaml`` inherits and overrides."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    cfg["_config_path"] = str(path)
    validate(cfg)
    return cfg


def validate(cfg: Dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    m = cfg["matrix"]
    if m["primary_model"] not in cfg["models"]:
        raise ValueError(f"matrix.primary_model {m['primary_model']!r} is not defined under models")
    for r in m["ratios"]:
        if not 0.0 < float(r) < 1.0:
            raise ValueError(f"ratio {r} must be in (0, 1)")
    bad = [x for x in m["methods"] if x not in METHODS]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {METHODS}")
    if cfg["runtime"]["dtype"] not in ("bfloat16", "float16"):
        raise ValueError("runtime.dtype must be bfloat16 or float16 (magnitude pruning uses 16-bit patterns)")
    if cfg["pruning"]["layers"]["bi_variant"] not in ("frobenius", "cosine"):
        raise ValueError("pruning.layers.bi_variant must be 'frobenius' or 'cosine'")


def pct(ratio: float) -> int:
    return int(round(float(ratio) * 100))


# --------------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Condition:
    no: int                      # running number (equals thesis Table 17 for the full matrix)
    id: str                      # unique, filesystem-safe, e.g. "7b-heads-30-lora"
    model_key: str               # key under ``models`` in the config
    method: str                  # "none" (baseline) | magnitude | heads | layers
    pct: int                     # nominal pruning ratio in percent (0 for the baseline)
    lora_rank: Optional[int]     # None = no recovery
    evals: Tuple[str, ...]       # subset of EVAL_KINDS
    efficiency: bool             # measure memory / throughput
    group: str = ""              # which Table 17 row family it belongs to
    enabled: bool = True

    @property
    def recipe_key(self) -> Optional[str]:
        """Recipe shared by the plain and the LoRA-recovered condition."""
        return None if self.method == "none" else f"{self.model_key}-{self.method}-{self.pct}"

    @property
    def is_noop(self) -> bool:
        return not self.evals and not self.efficiency

    @property
    def ratio(self) -> float:
        return self.pct / 100.0


def _hit(rule: Any, p: int) -> bool:
    if rule in (None, False, [], ()):
        return False
    if rule is True or rule == "all":
        return True
    return p in {int(x) for x in rule}


def _evals_for(rules: Dict[str, Any], p: int) -> Tuple[str, ...]:
    return tuple(k for k in EVAL_KINDS if _hit(rules.get(k), p))


def expand_conditions(cfg: Dict[str, Any]) -> List[Condition]:
    """Expand ``cfg['matrix']`` into the ordered list of conditions (thesis Table 17)."""
    m = cfg["matrix"]
    models = cfg["models"]
    primary = m["primary_model"]
    lora_rank = int(cfg["lora"]["rank"])
    ratios = [float(r) for r in m["ratios"]]
    out: List[Condition] = []
    n = 1  # running number of the next numbered condition (Table 17 starts pruned conditions at 1)

    def add(no=None, **kw):
        nonlocal n
        if no is None:
            no = n
            n += 1
        out.append(Condition(no=no, **kw))

    # No. 0: baselines (Table 17 lists the 7B and 13B baselines together as row 0)
    for role, key in (("primary", primary), ("secondary", m.get("secondary_model"))):
        if key and key in models:
            spec = m.get("baseline", {}).get(role, {})
            add(no=0, id=f"{key}-base", model_key=key, method="none", pct=0, lora_rank=None,
                evals=tuple(k for k in EVAL_KINDS if k in spec.get("evals", [])),
                efficiency=bool(spec.get("efficiency", False)), group="baseline")

    # Nos. 1-36: primary model, method x {plain, LoRA}
    for method in m["methods"]:
        for recovered in (False, True):
            rules = m["lora" if recovered else "plain"]
            for r in ratios:
                p = pct(r)
                add(id=f"{primary}-{method}-{p}{'-lora' if recovered else ''}",
                    model_key=primary, method=method, pct=p,
                    lora_rank=lora_rank if recovered else None,
                    evals=_evals_for(rules, p), efficiency=_hit(rules.get("efficiency"), p),
                    group=f"{method}{'+lora' if recovered else ''}")

    # Nos. 37-40: cross-scale check on the secondary model
    cs = m.get("cross_scale")
    if cs and cs["model"] in models:
        for r in cs["ratios"]:
            p = pct(r)
            add(id=f"{cs['model']}-{cs['method']}-{p}", model_key=cs["model"], method=cs["method"],
                pct=p, lora_rank=None, evals=_evals_for(cs, p), efficiency=_hit(cs.get("efficiency"), p),
                group="cross-scale")
        lo = cs.get("lora") or {}
        for r in lo.get("ratios", []):
            p = pct(r)
            add(id=f"{cs['model']}-{cs['method']}-{p}-lora", model_key=cs["model"], method=cs["method"],
                pct=p, lora_rank=lora_rank, evals=_evals_for(lo, p), efficiency=False,
                group="cross-scale+lora", enabled=bool(lo.get("enabled", False)))

    # No. 41: LoRA-rank ablation (the default rank is the main LoRA condition)
    ab = m.get("rank_ablation")
    if ab:
        p = pct(ab["ratio"])
        ab_no = n
        n += 1
        for rank in ab["ranks"]:
            if int(rank) == lora_rank:
                continue
            add(no=ab_no, id=f"{primary}-{ab['method']}-{p}-lora-r{int(rank)}", model_key=primary, method=ab["method"],
                pct=p, lora_rank=int(rank), evals=tuple(k for k in EVAL_KINDS if k in ab.get("evals", [])),
                efficiency=False, group="rank-ablation")

    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate condition ids: " + ", ".join(sorted({i for i in ids if ids.count(i) > 1})))
    return out


def select_conditions(
    conds: Sequence[Condition],
    only: Optional[Iterable[str]] = None,
    shard: Optional[Tuple[int, int]] = None,
    include_disabled: bool = False,
) -> List[Condition]:
    """Filter by id / glob / running number and by shard ``(k, n)`` (condition i runs on shard i % n == k)."""
    sel = list(conds)
    if not include_disabled:
        sel = [c for c in sel if c.enabled]
    if only:
        pats = [str(o) for o in only]
        sel = [c for c in sel if any(fnmatch.fnmatch(c.id, p) or str(c.no) == p for p in pats)]
    if shard is not None:
        k, n = shard
        if not (n >= 1 and 0 <= k < n):
            raise ValueError(f"invalid shard {k}/{n}")
        sel = [c for i, c in enumerate(sel) if i % n == k]
    return sel


def parse_shard(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    k, n = text.split("/")
    return int(k), int(n)


# --------------------------------------------------------------------------- paths
def recipe_path(cfg: Dict[str, Any], cond: Condition) -> Optional[Path]:
    return None if cond.recipe_key is None else Path(cfg["paths"]["recipes"]) / f"{cond.recipe_key}.json"


def adapter_dir(cfg: Dict[str, Any], cond: Condition) -> Path:
    return Path(cfg["paths"]["adapters"]) / cond.id


def result_dir(cfg: Dict[str, Any], cond: Condition) -> Path:
    return Path(cfg["paths"]["results"]) / cond.id


def eval_path(cfg: Dict[str, Any], cond: Condition, kind: str) -> Path:
    return result_dir(cfg, cond) / f"eval_{kind}.json"


def efficiency_path(cfg: Dict[str, Any], cond: Condition, setting: str) -> Path:
    return result_dir(cfg, cond) / f"efficiency_{setting}.json"


def expected_outputs(cfg: Dict[str, Any], cond: Condition) -> List[Path]:
    """Files whose existence means the condition is finished (used for resuming and ``status``)."""
    if cond.is_noop:
        return []
    outs: List[Path] = []
    if cond.recipe_key:
        outs.append(recipe_path(cfg, cond))
    if cond.lora_rank:
        outs.append(adapter_dir(cfg, cond) / "train_summary.json")
    outs += [eval_path(cfg, cond, k) for k in cond.evals]
    if cond.efficiency:
        outs += [efficiency_path(cfg, cond, s) for s in cfg["efficiency"]["settings"]]
    return outs


def describe(conds: Sequence[Condition]) -> str:
    """Plain-text plan of the matrix (used by ``--dry-run``)."""
    head = f"{'No':>3}  {'id':<28} {'method':<10} {'ratio':>5}  {'LoRA':>4}  {'evals':<16} eff  note"
    rows = [head, "-" * len(head)]
    for c in conds:
        note = "disabled" if not c.enabled else ("nothing reported in Table 17" if c.is_noop else "")
        rows.append(
            f"{c.no:>3}  {c.id:<28} {c.method:<10} {c.pct:>4}%  "
            f"{('r=' + str(c.lora_rank)) if c.lora_rank else '-':>4}  {','.join(c.evals) or '-':<16} "
            f"{'yes' if c.efficiency else ' - '}  {note}"
        )
    return "\n".join(rows)
