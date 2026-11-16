"""Summary statistics over replicate runs with different random seeds (pure Python, no PyTorch).

The thesis reports a single run (seed 42), which says little about run-to-run variation. Replicates give
means, standard deviations, confidence intervals and tests. Every replicate is an ordinary run of the pipeline with its own seed and its own
output folders, described by a small config that inherits from the main one:

    # configs/replicates/seed43.yaml
    extends: ../full.yaml
    seed: 43
    paths: {recipes: recipes_seed43, adapters: adapters_seed43, results: results_seed43}

``llmprune aggregate --configs configs/full.yaml configs/replicates/seed43.yaml ...`` then reads the result
files of all of them. What the seed changes: the C4 calibration sample (head scores, Block Influence and
therefore which heads / layers are removed) and the LoRA data order and initialisation. Magnitude pruning
without recovery is deterministic, so its spread is exactly zero by construction.
"""
from __future__ import annotations

import csv
import io
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import efficiency_path, eval_path, expand_conditions, load_config
from .io_utils import read_json

# Two-sided 95 % critical values of Student's t distribution, df = 1 ... 30.
_T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t_critical_95(df: int) -> float:
    """Two-sided 95 % critical value of Student's t (table up to 30 degrees of freedom, then 1.96 + 2.4/df)."""
    if df < 1:
        raise ValueError("need at least one degree of freedom")
    return _T95[df - 1] if df <= 30 else 1.96 + 2.4 / df


def _betacf(a: float, b: float, x: float, max_iter: int = 500, eps: float = 3e-14) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz algorithm)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h  # pragma: no cover - converges in a few dozen iterations for every argument used here


def _regularised_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """P(|T| >= |t|) for Student's t with ``df`` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return _regularised_beta(df / 2.0, 0.5, df / (df + t * t))


def summarise(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """n, mean, sample standard deviation and the 95 % confidence interval of the mean (needs n >= 2)."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else None
    half = t_critical_95(n - 1) * sd / math.sqrt(n) if sd is not None else None
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": None if half is None else mean - half,
            "ci95_high": None if half is None else mean + half, "min": min(values), "max": max(values)}


def paired_test(a: Dict[int, float], b: Dict[int, float]) -> Dict[str, Optional[float]]:
    """Paired t-test of ``a - b`` over the seeds that both have, with the 95 % CI of the difference and Cohen's d_z."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    n = len(diffs)
    out: Dict[str, Optional[float]] = {"n": n, "mean_diff": None, "sd_diff": None, "t": None, "df": n - 1 if n else None,
                                       "p_two_sided": None, "ci95_low": None, "ci95_high": None, "cohens_dz": None}
    if n == 0:
        return out
    out["mean_diff"] = statistics.fmean(diffs)
    if n < 2:
        return out
    sd = statistics.stdev(diffs)
    half = t_critical_95(n - 1) * sd / math.sqrt(n)
    out.update(sd_diff=sd, ci95_low=out["mean_diff"] - half, ci95_high=out["mean_diff"] + half)
    if sd > 0:
        out["t"] = out["mean_diff"] / (sd / math.sqrt(n))
        out["p_two_sided"] = t_two_sided_p(out["t"], n - 1)
        out["cohens_dz"] = out["mean_diff"] / sd
    else:  # identical differences in every replicate (e.g. deterministic methods)
        out["t"] = 0.0 if out["mean_diff"] == 0 else math.inf
        out["p_two_sided"] = 1.0 if out["mean_diff"] == 0 else 0.0
    return out


# --------------------------------------------------------------------------- reading result files
def metric_files(cfg: Dict[str, Any], cond) -> List[Tuple[str, Path, str]]:
    """(metric name, result file, JSON key) for everything the condition can have reported."""
    out: List[Tuple[str, Path, str]] = []
    for kind, key in (("ppl", "wikitext2_ppl"), ("mmlu", "macro_average_pct"), ("glue", "average_pct")):
        if kind in cond.evals:
            out.append((kind, eval_path(cfg, cond, kind), key))
    if cond.efficiency:
        for setting in cfg["efficiency"]["settings"]:
            path = efficiency_path(cfg, cond, setting)
            out += [(f"tokens_per_s[{setting}]", path, "tokens_per_s_mean"),
                    (f"weights_gb[{setting}]", path, "weights_allocated_gb"),
                    (f"peak_gb[{setting}]", path, "peak_allocated_gb")]
    return out


def collect(config_paths: Sequence[str]) -> Tuple[List[Tuple[int, str]], Dict[Tuple[str, str], Dict[int, float]], List[int]]:
    """Read the results of every replicate config.

    Returns (conditions in matrix order, {(condition id, metric): {seed: value}}, seeds).
    """
    cfgs = [load_config(p) for p in config_paths]
    seeds = [int(c["seed"]) for c in cfgs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"the replicate configs must use different seeds, got {seeds}")
    folders = [str(Path(c["paths"]["results"]).resolve()) for c in cfgs]
    if len(set(folders)) != len(folders):
        raise ValueError("the replicate configs must write to different results folders")
    order: List[Tuple[int, str]] = []
    values: Dict[Tuple[str, str], Dict[int, float]] = {}
    for cfg in cfgs:
        for cond in expand_conditions(cfg):
            if not cond.enabled:
                continue
            if (cond.no, cond.id) not in order:
                order.append((cond.no, cond.id))
            for metric, path, key in metric_files(cfg, cond):
                if not path.is_file():
                    continue
                v = read_json(path).get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    values.setdefault((cond.id, metric), {})[int(cfg["seed"])] = float(v)
    return order, values, seeds


# --------------------------------------------------------------------------- report
def _f(x: Optional[float], digits: int = 3) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def write_replicate_report(config_paths: Sequence[str], out_dir, compare: Optional[Sequence[str]] = None,
                           metric: str = "ppl") -> List[Path]:
    """Write ``replicates.csv`` / ``REPLICATES.md`` (and ``paired.csv`` when ``compare`` = [id_a, id_b] is given)."""
    order, values, seeds = collect(config_paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = ["no", "condition", "metric", "n", "mean", "sd", "ci95_low", "ci95_high", "min", "max", "seeds"]
    rows: List[List[str]] = []
    for no, cid in order:
        for (cond_id, name), per_seed in values.items():
            if cond_id != cid:
                continue
            s = summarise(list(per_seed.values()))
            rows.append([str(no), cid, name, str(s["n"]), _f(s["mean"], 4), _f(s["sd"], 4), _f(s["ci95_low"], 4),
                         _f(s["ci95_high"], 4), _f(s["min"], 4), _f(s["max"], 4), " ".join(str(k) for k in sorted(per_seed))])
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    written = [out / "replicates.csv"]
    written[0].write_text(buf.getvalue(), encoding="utf-8")

    md = ["# Replicates", "",
          f"Seeds: {', '.join(str(s) for s in seeds)}. Mean, sample standard deviation (n - 1) and the 95 % confidence "
          "interval of the mean (Student's t). With n < 2 the spread is undefined and left empty; with very few seeds "
          "the interval is wide, which is the honest statement about what a handful of runs can show.", "",
          "| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if compare:
        a_id, b_id = compare
        a, b = values.get((a_id, metric), {}), values.get((b_id, metric), {})
        t = paired_test(a, b)
        keys = list(t)
        pbuf = io.StringIO()
        pw = csv.writer(pbuf, lineterminator="\n")
        pw.writerow(["a", "b", "metric"] + keys)
        pw.writerow([a_id, b_id, metric] + [_f(t[k], 6) if isinstance(t[k], float) else ("" if t[k] is None else str(t[k])) for k in keys])
        (out / "paired.csv").write_text(pbuf.getvalue(), encoding="utf-8")
        written.append(out / "paired.csv")
        md += ["", f"## Paired comparison on {metric}: {a_id} - {b_id}", "",
               "Paired t-test over the seeds both conditions share (differences of the same seed). "
               + ("; ".join(f"{k} = {_f(t[k], 4) if isinstance(t[k], float) else t[k]}" for k in keys))]
    (out / "REPLICATES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(out / "REPLICATES.md")
    return written
