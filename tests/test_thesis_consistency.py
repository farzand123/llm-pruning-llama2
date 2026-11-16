"""Internal consistency of the numbers printed in the thesis, recomputed with this repository's formulas.

If one of these fails, the thesis table (not the code) contains an inconsistency.
"""
import pytest

from llmprune import metrics as M
from llmprune.report import thesis_reported

T = thesis_reported()
METHODS = ("magnitude", "heads", "layers")


@pytest.mark.parametrize("method", METHODS)
def test_table23_recovery_ratios_follow_from_tables_18_to_20(method):
    ppl, glue, mmlu, rec = T["ppl"], T["glue_30"], T["mmlu"], T["recovery"][method]
    b_ppl, b_mmlu, b_glue = T["baseline"]["ppl"], T["baseline"]["mmlu"], T["baseline"]["glue"]["average"]
    for p in (30, 50, 70):
        r = M.recovery_ratio(b_ppl, ppl[method][str(p)], ppl[f"{method}_lora"][str(p)])
        assert r == pytest.approx(rec[f"ppl_{p}"], abs=0.011)
    r = M.recovery_ratio(b_glue, glue[method]["average"], glue[f"{method}_lora"]["average"], True)
    assert r == pytest.approx(rec["glue_avg_30"], abs=0.011)
    r = M.recovery_ratio(b_mmlu, mmlu[method]["50"], mmlu[method]["50_lora"], True)
    assert r == pytest.approx(rec["mmlu_50"], abs=0.011)


def test_glue_averages_are_means_of_the_four_tasks():
    rows = dict(T["glue_30"], baseline=T["baseline"]["glue"])
    for name, d in rows.items():
        mean = sum(d[k] for k in ("sst2", "mnli", "qqp", "qnli")) / 4
        assert mean == pytest.approx(d["average"], abs=0.051), name


def test_table22_speedups_follow_from_throughput():
    e = T["efficiency_30"]
    for key, base in (("7b-magnitude", "7b-none"), ("7b-heads", "7b-none"), ("7b-layers", "7b-none"), ("13b-heads", "13b-none")):
        assert M.speedup(e[key]["throughput_tok_s"], e[base]["throughput_tok_s"]) == pytest.approx(e[key]["speedup"], abs=0.006)


def test_table24_rank_selection_is_16():
    ab = T["rank_ablation"]
    r, _ = M.choose_rank({int(k): v["ppl"] for k, v in ab.items()}, {int(k): v["mmlu"] for k, v in ab.items()})
    assert r == 16


def test_main_lora_run_equals_rank16_ablation_row():
    assert T["ppl"]["heads_lora"]["50"] == T["rank_ablation"]["16"]["ppl"]
    assert T["mmlu"]["heads"]["50_lora"] == T["rank_ablation"]["16"]["mmlu"]
