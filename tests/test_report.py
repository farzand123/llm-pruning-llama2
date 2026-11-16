"""End-to-end test of the report: feed the thesis's own numbers in as 're-run results' and check that
the regenerated tables equal the thesis tables (and every delta is zero)."""
from pathlib import Path

import numpy as np
import pytest

from llmprune import config as K
from llmprune import counting as C
from llmprune import report as R
from llmprune import selection as sel
from llmprune.io_utils import write_json

ROOT = Path(__file__).resolve().parents[1]
T = R.thesis_reported()
A7 = C.LLAMA2_7B


def _ppl(c):
    if c.method == "none":
        return T["baseline"]["ppl"]
    if c.group == "rank-ablation":
        return T["rank_ablation"][str(c.lora_rank)]["ppl"]
    return T["ppl"][c.method + ("_lora" if c.lora_rank else "")][str(c.pct)]


def _mmlu(c):
    if c.method == "none":
        return T["baseline"]["mmlu"]
    if c.group == "rank-ablation":
        return T["rank_ablation"][str(c.lora_rank)]["mmlu"]
    return T["mmlu"][c.method][f"{c.pct}_lora" if c.lora_rank else str(c.pct)]


def _glue(c):
    return T["baseline"]["glue"] if c.method == "none" else T["glue_30"][c.method + ("_lora" if c.lora_rank else "")]


def _recipe(method, p):
    rng = np.random.default_rng(p)
    base = C.total_params(A7)
    r = {"model_key": "7b", "method": method, "arch": {"hidden": 4096, "ffn": 11008, "layers": 32, "heads": 32,
                                                       "vocab": 32000, "kv_heads": None, "tied_embeddings": False},
         "params_before": {"structural": base, "nonzero": base}}
    if method == "heads":
        remove, info = sel.select_heads_to_remove(rng.random((32, 32)), p / 100)
        keep = sel.keep_lists(remove)
        hpl = [len(k) for k in keep]
        total = C.total_params(A7, hpl)
        r.update(heads_keep=keep, head_selection=info, params_after={"structural": total, "nonzero": total},
                 shapes_before={"n_heads_total": 1024, "layers": [{"q_proj": [4096, 4096], "o_proj": [4096, 4096]}] * 32,
                                "attention_params": 32 * C.attention_params(A7, 32), "ffn_params": 32 * C.ffn_params(A7)},
                 shapes_after={"n_heads_total": sum(hpl), "layers": [{"q_proj": [128 * h, 4096], "o_proj": [4096, 128 * h]} for h in hpl],
                               "attention_params": sum(C.attention_params(A7, h) for h in hpl), "ffn_params": 32 * C.ffn_params(A7)})
    elif method == "layers":
        n = int(32 * p / 100 + 1e-9)
        bi = rng.random(32)
        total = C.total_params(A7, n_layers=32 - n)
        r.update(layers_removed=sel.select_layers_to_remove(bi, p / 100), bi_variant="frobenius",
                 bi_scores={"frobenius": bi.tolist(), "cosine": bi.tolist()}, params_after={"structural": total, "nonzero": total})
    else:
        r.update(params_after={"structural": base, "nonzero": base - int(C.prunable_params(A7) * p / 100)})
    return r


@pytest.fixture()
def populated(tmp_path):
    cfg = K.load_config(ROOT / "configs" / "full.yaml")
    cfg["paths"] = {"recipes": str(tmp_path / "recipes"), "adapters": str(tmp_path / "adapters"), "results": str(tmp_path / "results")}
    for c in K.expand_conditions(cfg):
        if c.is_noop or not c.enabled:
            continue
        if "ppl" in c.evals:
            write_json(K.eval_path(cfg, c, "ppl"), {"wikitext2_ppl": _ppl(c)})
        if "mmlu" in c.evals:
            write_json(K.eval_path(cfg, c, "mmlu"), {"macro_average_pct": _mmlu(c)})
        if "glue" in c.evals:
            g = _glue(c)
            write_json(K.eval_path(cfg, c, "glue"), {"acc_pct": {k: g[k] for k in ("sst2", "mnli", "qqp", "qnli")},
                                                      "average_pct": g["average"], "protocol": {"num_fewshot": 0, "tasks": [], "limit": None}})
        if c.efficiency:
            key = f"{c.model_key}-{c.method if c.method != 'none' else 'none'}"
            e = T["efficiency_30"][key]
            write_json(K.efficiency_path(cfg, c, "single"), {"tokens_per_s_mean": e["throughput_tok_s"], "tokens_per_s_std": 1.5,
                                                              "weights_allocated_gb": e["analytic_weight_gb"], "peak_allocated_gb": e["reported_gpu_memory_gb"]})
        if c.recipe_key and c.model_key == "7b":
            write_json(K.recipe_path(cfg, c), _recipe(c.method, c.pct))
    return cfg


def _row(table, label):
    return next(r for r in table.rows if r[0].startswith(label))


def test_all_tables_build(populated):
    res = R.Results(populated)
    assert len(R.build_tables(res)) == 10


def test_table23_equals_thesis(populated):
    t = R.table_recovery(R.Results(populated))
    for method, label in (("magnitude", "Magnitude"), ("heads", "Attention-head"), ("layers", "Layer")):
        row = _row(t, label)
        thesis = T["recovery"][method]
        vals = [thesis[k] for k in ("ppl_30", "ppl_50", "ppl_70", "glue_avg_30", "mmlu_50")]
        assert [float(x) for x in row[1:]] == pytest.approx(vals, abs=0.011)


def test_table24_marks_rank_16_and_matches_parameter_counts(populated):
    t = R.table_rank_ablation(R.Results(populated))
    assert [r[0] for r in t.rows] == ["4", "8", "16 (selected)", "32", "64"]
    assert [r[3] for r in t.rows] == ["3.15 million", "6.29 million", "12.58 million", "25.17 million", "50.33 million"]
    assert "r* = 16" in t.note


def test_table21_and_22_match_thesis(populated):
    res = R.Results(populated)
    t21 = R.table_params(res)
    assert _row(t21, "Baseline")[1:3] == ["6.738", "6.738"]
    assert _row(t21, "Magnitude")[1:4] == ["6.738", "4.796", "28.8%"]
    assert _row(t21, "Attention-head")[1:4] == ["6.095", "6.095", "9.6%"]
    assert _row(t21, "Layer")[1:4] == ["4.917", "4.917", "27.0%"]
    t22 = R.table_efficiency(res)
    heads, layer, mag = _row(t22, "Attention-head 7B"), _row(t22, "Layer 7B"), _row(t22, "Magnitude 7B")
    assert heads[1] == "11.3%" and layer[1] == "27.6%" and mag[1] == "0% (dense)"
    assert heads[2] == "12.19" and layer[2] == "9.83"
    col = t22.header.index("[single] speedup")
    assert heads[col] == "1.38×" and layer[col] == "1.42×" and mag[col] == "1.02×"
    assert _row(t22, "Attention-head 13B")[col] == "1.38×"
    assert heads[t22.header.index("[auto] speedup")] == "–"      # no multi-GPU results were written


def test_comparison_with_thesis_has_zero_deltas(populated):
    rows = R.compare_with_thesis(R.Results(populated))
    have = [r for r in rows if r["rerun"] is not None]
    assert len(have) == 110   # 7 baseline + 36 PPL + 30 GLUE + 21 MMLU + 10 rank ablation + 6 throughput
    assert all(abs(r["delta"]) < 1e-9 for r in have)


def test_heads_per_layer_and_layers_removed_tables(populated):
    res = R.Results(populated)
    t15 = R.table_heads_per_layer(res)
    assert t15.rows[2][0] == "30% (307 heads removed)"
    assert len(t15.rows[2][1].split(", ")) == 32
    t16 = R.table_layers_removed(res)
    assert t16.rows[2][1] == "9" and t16.rows[2][2] == "28.1%"
    t14 = R.table_shapes(res)
    assert t14.rows[0][1:] == ["1024", "717"]


def test_write_report_creates_files(populated, tmp_path):
    out = tmp_path / "report"
    written = R.write_report(populated, out)
    names = {p.name for p in written}
    assert {"REPORT.md", "comparison_with_thesis.csv", "table18_ppl.csv", "table24_rank.csv"} <= names
    assert "Table 23" in (out / "REPORT.md").read_text(encoding="utf-8")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    assert (out / "figures" / "fig8_ppl_vs_ratio.png").stat().st_size > 1000


def test_empty_results_still_produce_a_report(tmp_path):
    cfg = K.load_config(ROOT / "configs" / "full.yaml")
    cfg["paths"] = {"recipes": str(tmp_path / "r"), "adapters": str(tmp_path / "a"), "results": str(tmp_path / "x")}
    written = R.write_report(cfg, tmp_path / "out")
    assert any(p.name == "REPORT.md" for p in written)
    assert "–" in R.table_ppl(R.Results(cfg)).to_markdown()
