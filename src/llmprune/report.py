"""Turn the result tree into the thesis tables (14-24), figures (8-12) and a thesis-vs-re-run
comparison. Pure Python (matplotlib is optional and only needed for figures).

Table numbers refer to the revised thesis. Every cell is computed from files written by
``llmprune run``; a missing file shows as an en dash, so partial runs still produce a report.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import counting as C
from . import metrics as M
from .config import (Condition, eval_path, efficiency_path, expand_conditions, pct)
from .io_utils import read_json

METHOD_LABEL = {"magnitude": "Magnitude", "heads": "Attention-head", "layers": "Layer"}
_KNOWN_ARCH = {"7b": C.LLAMA2_7B, "13b": C.LLAMA2_13B}


def thesis_reported() -> Dict[str, Any]:
    """Numbers exactly as printed in the thesis (Tables 18-24)."""
    with open(Path(__file__).parent / "resources" / "thesis_reported.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- results access
class Results:
    """Read-only view of ``paths.results`` / ``paths.recipes`` / ``paths.adapters``."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.conds = expand_conditions(cfg)
        self.by_id = {c.id: c for c in self.conds}
        m = cfg["matrix"]
        self.primary: str = m["primary_model"]
        self.secondary: Optional[str] = m.get("secondary_model")
        self.methods: List[str] = list(m["methods"])
        self.ratios: List[int] = [pct(r) for r in m["ratios"]]
        plain, lora = m["plain"], m["lora"]
        self.glue_pct = _first(plain.get("glue"), 30)
        self.eff_pct = _first(plain.get("efficiency"), 30)
        self.mmlu_lora_pct = _first(lora.get("mmlu"), 50)
        self.recipes_dir = Path(cfg["paths"]["recipes"])

    # -- ids
    def cid(self, method: str, p: int, lora: bool = False, model: Optional[str] = None) -> str:
        key = model or self.primary
        return f"{key}-base" if method == "none" else f"{key}-{method}-{p}{'-lora' if lora else ''}"

    # -- raw reads
    @staticmethod
    def _read(path: Path) -> Optional[dict]:
        return read_json(path) if Path(path).is_file() else None

    def eval(self, cid: str, kind: str) -> Optional[dict]:
        c = self.by_id.get(cid)
        return None if c is None else self._read(eval_path(self.cfg, c, kind))

    def ppl(self, cid: str) -> Optional[float]:
        d = self.eval(cid, "ppl")
        return None if d is None else d["wikitext2_ppl"]

    def mmlu(self, cid: str) -> Optional[float]:
        d = self.eval(cid, "mmlu")
        return None if d is None else d["macro_average_pct"]

    def glue(self, cid: str) -> Optional[Dict[str, float]]:
        d = self.eval(cid, "glue")
        return None if d is None else {**d["acc_pct"], "average": d["average_pct"]}

    def efficiency(self, cid: str, setting: str) -> Optional[dict]:
        c = self.by_id.get(cid)
        return None if c is None else self._read(efficiency_path(self.cfg, c, setting))

    def recipe(self, model: str, method: str, p: int) -> Optional[dict]:
        return self._read(self.recipes_dir / f"{model}-{method}-{p}.json")

    def train_summary(self, cid: str) -> Optional[dict]:
        c = self.by_id.get(cid)
        if c is None:
            return None
        return self._read(Path(self.cfg["paths"]["adapters"]) / c.id / "train_summary.json")

    def arch(self, model: str) -> Optional[C.Arch]:
        for method in ("heads", "layers", "magnitude"):
            for p in self.ratios:
                r = self.recipe(model, method, p)
                if r and r.get("arch"):
                    return C.Arch(**r["arch"])
        return _KNOWN_ARCH.get(model)


def _first(x, default):
    if isinstance(x, (list, tuple)) and x:
        return int(x[0])
    return default


# --------------------------------------------------------------------------- table object
@dataclass
class Table:
    key: str
    title: str
    header: List[str]
    rows: List[List[str]]
    note: str = ""

    def to_markdown(self) -> str:
        out = [f"### {self.title}", "", "| " + " | ".join(self.header) + " |",
               "| " + " | ".join(["---"] * len(self.header)) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        if self.note:
            out += ["", f"_{self.note}_"]
        return "\n".join(out) + "\n"

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.header)
        w.writerows(self.rows)
        return buf.getvalue()


def _label(method: str, lora: bool) -> str:
    return METHOD_LABEL[method] + (" + LoRA" if lora else "")


# --------------------------------------------------------------------------- tables
def table_ppl(res: Results) -> Table:
    rows = [[_label(m, lora)] + [M.fmt(res.ppl(res.cid(m, p, lora))) for p in res.ratios]
            for m in res.methods for lora in (False, True)]
    base = M.fmt(res.ppl(res.cid("none", 0)))
    return Table("table18_ppl", f"Table 18 – WikiText-2 perplexity by pruning method and ratio (unpruned baseline {base}; lower is better)",
                 ["Method"] + [f"{p}%" for p in res.ratios], rows)


def table_glue(res: Results) -> Table:
    p = res.glue_pct
    cols = ["sst2", "mnli", "qqp", "qnli", "average"]

    def row(label, g):
        return [label] + [M.fmt(None if g is None else g.get(c), 1) for c in cols]

    rows = [row("Baseline (unpruned)", res.glue(res.cid("none", 0)))]
    for m in res.methods:
        for lora in (False, True):
            rows.append(row(_label(m, lora), res.glue(res.cid(m, p, lora))))
    d = res.eval(res.cid("none", 0), "glue")
    note = ""
    if d:
        pr = d["protocol"]
        note = f"Protocol: {pr.get('num_fewshot')}-shot, tasks {pr.get('tasks')}, limit {pr.get('limit')}. Report exactly this in Chapter III.B.2."
    return Table("table19_glue", f"Table 19 – GLUE development-set accuracy (%) at the {p}% pruning ratio",
                 ["Method", "SST-2", "MNLI", "QQP", "QNLI", "Average"], rows, note)


def table_mmlu(res: Results) -> Table:
    lp = res.mmlu_lora_pct
    rows = []
    for m in res.methods:
        rows.append([METHOD_LABEL[m]] + [M.fmt(res.mmlu(res.cid(m, p)), 1) for p in res.ratios]
                    + [M.fmt(res.mmlu(res.cid(m, lp, True)), 1)])
    base = M.fmt(res.mmlu(res.cid("none", 0)), 1)
    return Table("table20_mmlu", f"Table 20 – MMLU 5-shot accuracy (%) (unpruned baseline {base}%)",
                 ["Method"] + [f"{p}%" for p in res.ratios] + [f"{lp}% + LoRA"], rows,
                 "Macro average over the 57 subjects.")


def _params(res: Results, method: str, p: int, arch: Optional[C.Arch]):
    """(structural, non-zero, source) – measured from the recipe if present, else analytical."""
    if method == "none":
        r = next((x for m in res.methods if (x := res.recipe(res.primary, m, p))), None)
        if r and r.get("params_before"):
            return r["params_before"]["structural"], r["params_before"]["nonzero"], "measured"
        return (C.total_params(arch), C.total_params(arch), "analytic") if arch else (None, None, "")
    r = res.recipe(res.primary, method, p)
    if r and r.get("params_after"):
        return r["params_after"]["structural"], r["params_after"]["nonzero"], "measured"
    if not arch:
        return None, None, ""
    total = C.total_params(arch)
    if method == "magnitude":
        return total, total - C.prunable_params(arch) * p // 100, "analytic"
    if method == "heads":
        kept = arch.layers * arch.heads - int(arch.layers * arch.heads * p / 100 + 1e-9)
        s = C.total_params(arch, C.heads_per_layer_from_total(arch, kept))
        return s, s, "analytic"
    n_rm = int(arch.layers * p / 100 + 1e-9)
    s = C.total_params(arch, n_layers=arch.layers - n_rm)
    return s, s, "analytic"


def table_params(res: Results) -> Table:
    p = res.eff_pct
    arch = res.arch(res.primary)
    b_struct, b_nz, _ = _params(res, "none", 0, arch)
    rows = [["Baseline", _b(b_struct), _b(b_nz), "–", "1.00", "1.00"]]
    for m in res.methods:
        s, nz, _ = _params(res, m, p, arch)
        if s is None or b_struct is None:
            rows.append([METHOD_LABEL[m], "–", "–", "–", "–", "–"])
            continue
        rows.append([METHOD_LABEL[m], _b(s), _b(nz), M.fmt(C.reduction_pct(b_nz, nz), 1, "%"),
                     M.fmt(C.compression_ratio(b_struct, s)), M.fmt(C.compression_ratio(b_nz, nz))])
    return Table("table21_params", f"Table 21 – Parameter accounting at the {p}% pruning ratio (billions)",
                 ["Method", "Structural parameters", "Non-zero parameters", "Non-zero reduction", "CR_struct", "CR_nz"], rows,
                 "Measured from the pruned model when its recipe exists, otherwise computed analytically.")


def _b(x) -> str:
    return "–" if x is None else f"{x / 1e9:.3f}"


def _analytic_efficiency(res: Results, model: str, method: str, p: int):
    arch = res.arch(model)
    if arch is None:
        return None
    if method == "none":
        return C.profile(arch)
    if method == "magnitude":
        return C.profile(arch)
    r = res.recipe(model, method, p)
    if method == "heads":
        if r and r.get("heads_keep"):
            hpl = [len(k) for k in r["heads_keep"]]
        else:
            kept = arch.layers * arch.heads - int(arch.layers * arch.heads * p / 100 + 1e-9)
            hpl = C.heads_per_layer_from_total(arch, kept)
        return C.profile(arch, hpl)
    n_rm = len(r["layers_removed"]) if r and r.get("layers_removed") is not None else int(arch.layers * p / 100 + 1e-9)
    return C.profile(arch, n_layers=arch.layers - n_rm)


def table_efficiency(res: Results) -> Table:
    p = res.eff_pct
    settings = list(res.cfg["efficiency"]["settings"])
    hdr = ["Method", "Analytical FLOPs reduction", "Analytical weight memory (GB)", "Parameters (billions)"]
    for s in settings:
        hdr += [f"[{s}] weights allocated (GB)", f"[{s}] peak allocated (GB)", f"[{s}] throughput (tokens/s)", f"[{s}] speedup"]
    rows = []
    entries = [(res.primary, "none")] + [(res.primary, m) for m in res.methods]
    if res.secondary:
        entries += [(res.secondary, "none"), (res.secondary, "heads")]
    for model, method in entries:
        cid = res.cid(method, p, False, model)
        base_cid = res.cid("none", 0, False, model)
        prof = _analytic_efficiency(res, model, method, p)
        label = ("Baseline" if method == "none" else METHOD_LABEL[method]) + f" {model.upper()}"
        row = [label,
               "–" if method == "none" or prof is None else (M.fmt(prof["flops_reduction_pct"], 1, "%") if method != "magnitude" else "0% (dense)"),
               M.fmt(prof["weight_gb_bf16"] if prof else None),
               _b(prof["structural_params"] if prof else None)]
        for s in settings:
            e, b = res.efficiency(cid, s), res.efficiency(base_cid, s)
            row += [M.fmt(e and e.get("weights_allocated_gb")), M.fmt(e and e.get("peak_allocated_gb")),
                    M.fmt(e and e.get("tokens_per_s_mean"), 1),
                    M.fmt(e and b and M.speedup(e["tokens_per_s_mean"], b["tokens_per_s_mean"]), 2, "×")]
        rows.append(row)
    return Table("table22_efficiency", f"Table 22 – Analytical and measured efficiency at the {p}% pruning ratio", hdr, rows,
                 "Analytical FLOPs from Equation 13 (context 2,048). 'single' = one GPU; 'auto' = device_map=auto over all visible GPUs.")


def table_recovery(res: Results) -> Table:
    base_ppl = res.ppl(res.cid("none", 0))
    base_glue = (res.glue(res.cid("none", 0)) or {}).get("average")
    base_mmlu = res.mmlu(res.cid("none", 0))
    rows = []
    for m in res.methods:
        row = [METHOD_LABEL[m]]
        for p in (30, 50, 70):
            a, b = res.ppl(res.cid(m, p)), res.ppl(res.cid(m, p, True))
            row.append(M.fmt(None if None in (base_ppl, a, b) else M.recovery_ratio(base_ppl, a, b)))
        ga, gb = (res.glue(res.cid(m, res.glue_pct)) or {}).get("average"), (res.glue(res.cid(m, res.glue_pct, True)) or {}).get("average")
        row.append(M.fmt(None if None in (base_glue, ga, gb) else M.recovery_ratio(base_glue, ga, gb, True)))
        ma, mb = res.mmlu(res.cid(m, res.mmlu_lora_pct)), res.mmlu(res.cid(m, res.mmlu_lora_pct, True))
        row.append(M.fmt(None if None in (base_mmlu, ma, mb) else M.recovery_ratio(base_mmlu, ma, mb, True)))
        rows.append(row)
    return Table("table23_recovery", "Table 23 – Recovery ratio R achieved by LoRA (Equation 14)",
                 ["Method", "PPL, 30%", "PPL, 50%", "PPL, 70%", f"GLUE average, {res.glue_pct}%", f"MMLU, {res.mmlu_lora_pct}%"], rows,
                 "R = 1: full recovery, R = 0: none. For accuracy the sign of Q is reversed.")


def table_rank_ablation(res: Results) -> Table:
    ab = res.cfg["matrix"].get("rank_ablation")
    if not ab:
        return Table("table24_rank", "Table 24 – LoRA-rank ablation", ["LoRA rank"], [], "not configured")
    p, method, main_rank = pct(ab["ratio"]), ab["method"], int(res.cfg["lora"]["rank"])
    ppl, mmlu, rows_data = {}, {}, []
    for rank in sorted(int(r) for r in ab["ranks"]):
        cid = res.cid(method, p, True) if rank == main_rank else f"{res.primary}-{method}-{p}-lora-r{rank}"
        a, b = res.ppl(cid), res.mmlu(cid)
        if a is not None:
            ppl[rank] = a
        if b is not None:
            mmlu[rank] = b
        ts = res.train_summary(cid)
        trainable = ts.get("trainable_params") if ts else None
        if trainable is None:
            arch, rec = res.arch(res.primary), res.recipe(res.primary, method, p)
            if arch and rec and rec.get("heads_keep"):
                trainable = C.lora_trainable_params(arch, rank, [len(k) for k in rec["heads_keep"]])
        rows_data.append((rank, a, b, trainable))
    r_star, _ = M.choose_rank(ppl, mmlu)
    rows = [[f"{r}" + (" (selected)" if r == r_star else ""), M.fmt(a), M.fmt(b, 1),
             "–" if t is None else f"{t / 1e6:.2f} million"] for r, a, b, t in rows_data]
    note = (f"Equation 16 selects r* = {r_star}." if r_star else "Equation 16 could not be evaluated (missing results).")
    return Table("table24_rank", f"Table 24 – LoRA-rank ablation for {p}% {METHOD_LABEL[method].lower()} pruning",
                 ["LoRA rank", "WikiText-2 PPL", "MMLU accuracy (%)", "Trainable parameters"], rows, note)


def table_heads_per_layer(res: Results) -> Table:
    rows = []
    for p in res.ratios:
        r = res.recipe(res.primary, "heads", p)
        if not r:
            rows.append([f"{p}%", "–", "–", "–"])
            continue
        info = r["head_selection"]
        rows.append([f"{p}% ({info['n_removed']} heads removed)",
                     ", ".join(str(k) for k in info["heads_kept_per_layer"]),
                     ", ".join(map(str, info["layers_where_min_rule_was_binding"])) or "none",
                     ", ".join(map(str, info["layers_emptied_by_pure_global_ranking"])) or "none"])
    return Table("table15_heads_per_layer", "Table 15 – Heads retained per layer after attention-head pruning",
                 ["Pruning ratio", "Heads retained in layers 0 to L-1", "Layers where the minimum-heads rule was binding",
                  "Layers a pure global ranking would have emptied"], rows)


def table_layers_removed(res: Results) -> Table:
    rows = []
    for p in res.ratios:
        r = res.recipe(res.primary, "layers", p)
        if not r:
            rows.append([f"{p}%", "–", "–", "–", "–"])
            continue
        n_total = len(r["bi_scores"][r["bi_variant"]])
        rm = r["layers_removed"]
        rows.append([f"{p}%", str(len(rm)), M.fmt(100.0 * len(rm) / n_total, 1, "%"),
                     _b((r.get("params_after") or {}).get("structural")), ", ".join(map(str, rm))])
    return Table("table16_layers_removed", "Table 16 – Transformer blocks removed by layer pruning",
                 ["Nominal ratio", "Blocks removed", "Realised block fraction", "Total parameters after pruning (B)", "Indices of removed blocks"], rows)


def table_shapes(res: Results) -> Table:
    r = res.recipe(res.primary, "heads", res.eff_pct)
    rows = []
    if r and r.get("shapes_before") and r.get("shapes_after"):
        b, a = r["shapes_before"], r["shapes_after"]
        rows = [
            ["Heads in the model", str(b["n_heads_total"]), str(a["n_heads_total"])],
            ["W_Q in layer 0", "x".join(map(str, b["layers"][0]["q_proj"])), "x".join(map(str, a["layers"][0]["q_proj"]))],
            ["W_O in layer 0", "x".join(map(str, b["layers"][0]["o_proj"])), "x".join(map(str, a["layers"][0]["o_proj"]))],
            ["Attention-projection parameters", _b(b["attention_params"]), _b(a["attention_params"])],
            ["FFN parameters", _b(b["ffn_params"]), _b(a["ffn_params"])],
            ["Total parameters", _b(r["params_before"]["structural"]), _b(r["params_after"]["structural"])],
        ]
    return Table("table14_shapes", f"Table 14 – Tensor dimensions and parameter counts before and after {res.eff_pct}% attention-head pruning",
                 ["Quantity", "Before pruning", "After pruning"], rows)


def build_tables(res: Results) -> List[Table]:
    return [table_shapes(res), table_heads_per_layer(res), table_layers_removed(res), table_ppl(res), table_glue(res),
            table_mmlu(res), table_params(res), table_efficiency(res), table_recovery(res), table_rank_ablation(res)]


# --------------------------------------------------------------------------- thesis comparison
def compare_with_thesis(res: Results) -> List[dict]:
    """Every number the thesis printed next to the value this repository produced."""
    t = thesis_reported()
    rows: List[dict] = []

    def add(q, thesis, new):
        delta = None if (new is None or thesis is None) else new - thesis
        rows.append({"quantity": q, "thesis": thesis, "rerun": new, "delta": delta})

    add("Baseline WikiText-2 PPL", t["baseline"]["ppl"], res.ppl(res.cid("none", 0)))
    add("Baseline MMLU (%)", t["baseline"]["mmlu"], res.mmlu(res.cid("none", 0)))
    g = res.glue(res.cid("none", 0))
    for k, v in t["baseline"]["glue"].items():
        add(f"Baseline GLUE {k} (%)", v, None if g is None else g.get(k if k != "average" else "average"))
    for key, per in t["ppl"].items():
        method, lora = key.replace("_lora", ""), key.endswith("_lora")
        for p, v in per.items():
            add(f"PPL {method}{' +LoRA' if lora else ''} {p}%", v, res.ppl(res.cid(method, int(p), lora)))
    for key, d in t["glue_30"].items():
        method, lora = key.replace("_lora", ""), key.endswith("_lora")
        gg = res.glue(res.cid(method, 30, lora))
        for k, v in d.items():
            add(f"GLUE {k} {method}{' +LoRA' if lora else ''} 30%", v, None if gg is None else gg.get(k))
    for method, d in t["mmlu"].items():
        for col, v in d.items():
            lora = col.endswith("_lora")
            p = int(col.replace("_lora", ""))
            add(f"MMLU {method}{' +LoRA' if lora else ''} {p}%", v, res.mmlu(res.cid(method, p, lora)))
    for r, d in t["rank_ablation"].items():
        main = int(res.cfg["lora"]["rank"])
        ab = res.cfg["matrix"].get("rank_ablation") or {}
        cid = f"{res.primary}-{ab.get('method', 'heads')}-50-lora" + ("" if int(r) == main else f"-r{r}")
        add(f"Rank ablation PPL r={r}", d["ppl"], res.ppl(cid))
        add(f"Rank ablation MMLU r={r}", d["mmlu"], res.mmlu(cid))
    for key, d in t["efficiency_30"].items():
        model, method = key.split("-")
        e = res.efficiency(res.cid(method, 30, False, model), "single")
        e_auto = res.efficiency(res.cid(method, 30, False, model), "auto")
        add(f"Throughput {key} (tokens/s, thesis=multi-GPU) vs 'auto'", d["throughput_tok_s"], e_auto and e_auto.get("tokens_per_s_mean"))
        add(f"Throughput {key} (tokens/s) vs 'single'", d["throughput_tok_s"], e and e.get("tokens_per_s_mean"))
    return rows


def comparison_csv(rows: Sequence[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["quantity", "thesis", "rerun", "delta"])
    for r in rows:
        w.writerow([r["quantity"], r["thesis"], "" if r["rerun"] is None else f"{r['rerun']:.4f}",
                    "" if r["delta"] is None else f"{r['delta']:+.4f}"])
    return buf.getvalue()


# --------------------------------------------------------------------------- figures (optional)
def make_figures(res: Results, out_dir: Path) -> List[Path]:
    """Figures 8-12. Returns the written files; returns [] if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    xs = res.ratios
    # Figure 8: perplexity vs ratio (log y)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in res.methods:
        for lora in (False, True):
            ys = [res.ppl(res.cid(m, p, lora)) for p in xs]
            if any(y is not None for y in ys):
                ax.plot([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None],
                        marker="o", linestyle="--" if lora else "-", label=_label(m, lora))
    ax.set_yscale("log"); ax.set_xlabel("Nominal pruning ratio (%)"); ax.set_ylabel("WikiText-2 perplexity")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    save(fig, "fig8_ppl_vs_ratio.png")

    # Figure 10: MMLU vs ratio (no recovery)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in res.methods:
        ys = [res.mmlu(res.cid(m, p)) for p in xs]
        if any(y is not None for y in ys):
            ax.plot([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None], marker="o", label=METHOD_LABEL[m])
    ax.axhline(25, color="grey", linestyle=":", label="chance (25%)")
    ax.set_xlabel("Nominal pruning ratio (%)"); ax.set_ylabel("MMLU accuracy (%)"); ax.legend(fontsize=7)
    save(fig, "fig10_mmlu_vs_ratio.png")

    # Figure 9: GLUE at the reported ratio
    labels, vals = ["Baseline"], [(res.glue(res.cid("none", 0)) or {}).get("average")]
    for m in res.methods:
        for lora in (False, True):
            labels.append(_label(m, lora)); vals.append((res.glue(res.cid(m, res.glue_pct, lora)) or {}).get("average"))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(vals)), [v if v is not None else 0 for v in vals])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("GLUE average accuracy (%)")
    save(fig, "fig9_glue.png")

    # Figures 11 and 12: memory and throughput at the efficiency ratio
    entries = [(res.primary, "none")] + [(res.primary, m) for m in res.methods]
    names, analytic, alloc, peak, tps, err = [], [], [], [], [], []
    for model, method in entries:
        prof = _analytic_efficiency(res, model, method, res.eff_pct)
        e = res.efficiency(res.cid(method, res.eff_pct, False, model), "single")
        names.append("Baseline" if method == "none" else METHOD_LABEL[method])
        analytic.append(prof["weight_gb_bf16"] if prof else 0)
        alloc.append((e or {}).get("weights_allocated_gb") or 0)
        peak.append((e or {}).get("peak_allocated_gb") or 0)
        tps.append((e or {}).get("tokens_per_s_mean") or 0)
        err.append((e or {}).get("tokens_per_s_std") or 0)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    w = 0.27
    for i, (lab, series) in enumerate((("analytical weights", analytic), ("allocated after load", alloc), ("peak during generation", peak))):
        ax.bar([j + (i - 1) * w for j in range(len(names))], series, w, label=lab)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("GB"); ax.legend(fontsize=7)
    save(fig, "fig11_memory.png")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(names)), tps, yerr=err, capsize=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("tokens / s (single GPU)")
    save(fig, "fig12_throughput.png")
    return written


# --------------------------------------------------------------------------- driver
def write_report(cfg: Dict[str, Any], out_dir) -> List[Path]:
    """Write tables (markdown + CSV), the thesis comparison, figures and REPORT.md."""
    res = Results(cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(res)
    written: List[Path] = []
    for t in tables:
        (out / f"{t.key}.csv").write_text(t.to_csv(), encoding="utf-8")
        written.append(out / f"{t.key}.csv")
    comp = compare_with_thesis(res)
    (out / "comparison_with_thesis.csv").write_text(comparison_csv(comp), encoding="utf-8")
    written.append(out / "comparison_with_thesis.csv")
    figs = make_figures(res, out / "figures")
    n_have = sum(1 for r in comp if r["rerun"] is not None)
    md = ["# Results report", "",
          f"Config: `{cfg.get('_config_path')}`. Re-run values available for {n_have} of {len(comp)} quantities printed in the thesis.", "",
          "> Numbers here come from a **re-run** with this repository's code. They are not expected to equal the thesis tables "
          "exactly (the original code and logs were lost); see `comparison_with_thesis.csv`.", ""]
    md += [t.to_markdown() for t in tables]
    if figs:
        md += ["### Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figs]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    written.append(out / "REPORT.md")
    return written + figs
