"""Derived metrics used in the thesis (pure Python)."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The 57 MMLU subjects. The thesis reports the average over subjects (macro average), so the
# aggregate is recomputed from the per-subject results instead of trusting a group value.
MMLU_SUBJECTS: Tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
)


def macro_average(values: Iterable[float]) -> float:
    v = list(values)
    if not v:
        raise ValueError("no values to average")
    return sum(v) / len(v)


def recovery_ratio(baseline: float, pruned: float, recovered: float, higher_is_better: bool = False) -> float:
    """Equation 14:  R = (Q_pruned - Q_recovered) / (Q_pruned - Q_baseline).

    Q is perplexity (lower is better) or accuracy with the sign reversed, so that for both
    kinds R = 1 means full recovery and R = 0 means none. Returns NaN when the pruned
    model is no worse than the baseline (the ratio is undefined then).
    """
    sign = -1.0 if higher_is_better else 1.0
    qb, qp, qr = sign * baseline, sign * pruned, sign * recovered
    denom = qp - qb
    if abs(denom) < 1e-12:
        return float("nan")
    return (qp - qr) / denom


def speedup(throughput_pruned: float, throughput_baseline: float) -> float:
    """Equation 12."""
    return throughput_pruned / throughput_baseline


def choose_rank(
    ppl_by_rank: Dict[int, float],
    mmlu_by_rank: Dict[int, float],
    ppl_tol: float = 0.2,
    mmlu_tol: float = 0.5,
) -> Tuple[Optional[int], List[dict]]:
    """Equation 16:  r* = min{ r : dPPL(r -> 2r) < ppl_tol  and  dMMLU(r -> 2r) < mmlu_tol }.

    dPPL(r -> 2r) is the perplexity *improvement* from doubling the rank (PPL(r) - PPL(2r));
    dMMLU(r -> 2r) is the accuracy improvement (MMLU(2r) - MMLU(r)). Only ranks whose double
    was also evaluated can be tested. Returns (r_star or None, per-rank table).
    """
    table = []
    r_star = None
    for r in sorted(ppl_by_rank):
        r2 = 2 * r
        if r2 not in ppl_by_rank or r not in mmlu_by_rank or r2 not in mmlu_by_rank:
            continue
        d_ppl = ppl_by_rank[r] - ppl_by_rank[r2]
        d_mmlu = mmlu_by_rank[r2] - mmlu_by_rank[r]
        ok = d_ppl < ppl_tol and d_mmlu < mmlu_tol
        table.append({"rank": r, "double": r2, "d_ppl": d_ppl, "d_mmlu": d_mmlu, "satisfies": ok})
        if ok and r_star is None:
            r_star = r
    return r_star, table


def is_close(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return False
    return abs(a - b) <= tol


def fmt(x: Optional[float], nd: int = 2, suffix: str = "") -> str:
    """Format a number for a table cell; None / NaN become an en dash."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}{suffix}"


# --------------------------------------------------------------------------- lm-evaluation-harness results
def get_metric(entry: Optional[dict], name: str = "acc") -> Optional[float]:
    """Value of ``name`` in one task entry of an lm-evaluation-harness result.

    The 0.4.x harness stores filtered metrics as ``"acc,none"``; older versions use plain ``"acc"``.
    """
    if not entry:
        return None
    for key in (f"{name},none", name):
        v = entry.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def summarise_mmlu(results: Dict[str, dict], groups: Optional[Dict[str, dict]] = None) -> dict:
    """MMLU summary from harness results: the thesis reports the *macro* average over the 57 subjects,
    so it is recomputed from the per-subject accuracies; the harness' own group value (a micro average
    weighted by question count) is stored next to it."""
    per_subject: Dict[str, float] = {}
    for subject in MMLU_SUBJECTS:
        for key in (f"mmlu_{subject}", f"hendrycksTest-{subject}"):
            v = get_metric(results.get(key), "acc")
            if v is not None:
                per_subject[subject] = 100.0 * v
                break
    missing = [s for s in MMLU_SUBJECTS if s not in per_subject]
    if missing:
        raise ValueError(f"MMLU results lack {len(missing)} of {len(MMLU_SUBJECTS)} subjects (e.g. {missing[:3]})")
    group_value = None
    for source in (groups or {}, results):
        v = get_metric(source.get("mmlu"), "acc")
        if v is not None:
            group_value = 100.0 * v
            break
    return {
        "macro_average_pct": macro_average(per_subject.values()),
        "harness_group_acc_pct": group_value,
        "n_subjects": len(per_subject),
        "per_subject_acc_pct": per_subject,
    }


def summarise_glue(results: Dict[str, dict], tasks: Sequence[str]) -> dict:
    """Accuracy of each GLUE task (percent) and their unweighted mean (the thesis' "average")."""
    acc: Dict[str, float] = {}
    for t in tasks:
        v = get_metric(results.get(t), "acc")
        if v is None:
            raise ValueError(f"no accuracy for GLUE task {t!r} in the harness results (found: {sorted(results)[:8]})")
        acc[t] = 100.0 * v
    return {"acc_pct": acc, "average_pct": macro_average(acc.values())}
