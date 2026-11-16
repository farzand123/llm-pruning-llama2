"""Replicate statistics (pure Python). Reference values are the classical t-table and exact closed forms."""
import math
from pathlib import Path

import pytest

from llmprune import config as C
from llmprune import replicates as R
from llmprune.cli import main
from llmprune.io_utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def test_t_distribution_p_values_against_closed_forms_and_the_table():
    assert R.t_two_sided_p(0.0, 5) == pytest.approx(1.0)
    assert R.t_two_sided_p(1.0, 1) == pytest.approx(0.5, abs=1e-12)                      # Cauchy: 1 - (2/pi) atan(1)
    assert R.t_two_sided_p(1.0, 2) == pytest.approx(1 - 1 / math.sqrt(3), abs=1e-12)      # exact for df = 2
    for df in (1, 2, 3, 4, 5, 10, 20, 30):                                                # critical values -> p = 0.05
        assert R.t_two_sided_p(R.t_critical_95(df), df) == pytest.approx(0.05, abs=6e-4), df
    assert R.t_two_sided_p(math.inf, 3) == 0.0
    assert R.t_two_sided_p(50.0, 8) < 1e-9


def test_critical_values_beyond_the_table_are_close_to_the_truth():
    assert R.t_critical_95(60) == pytest.approx(2.000, abs=0.002)
    assert R.t_critical_95(120) == pytest.approx(1.980, abs=0.002)
    with pytest.raises(ValueError):
        R.t_critical_95(0)


def test_summary_statistics():
    s = R.summarise([10.0, 12.0, 14.0])
    assert s["mean"] == 12.0 and s["sd"] == 2.0 and s["n"] == 3
    half = 4.303 * 2.0 / math.sqrt(3)
    assert s["ci95_low"] == pytest.approx(12.0 - half) and s["ci95_high"] == pytest.approx(12.0 + half)
    single = R.summarise([5.0])
    assert single["sd"] is None and single["ci95_low"] is None and single["mean"] == 5.0
    with pytest.raises(ValueError):
        R.summarise([])


def test_paired_test():
    a = {42: 10.0, 43: 12.0, 44: 14.0, 45: 11.0}
    b = {42: 9.0, 43: 10.5, 44: 12.0, 45: 10.2}
    t = R.paired_test(a, b)
    d = [1.0, 1.5, 2.0, 0.8]
    mean = sum(d) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 3)
    assert t["n"] == 4 and t["df"] == 3 and t["mean_diff"] == pytest.approx(mean)
    assert t["t"] == pytest.approx(mean / (sd / 2)) and t["cohens_dz"] == pytest.approx(mean / sd)
    assert 0 < t["p_two_sided"] < 0.05 and t["ci95_low"] > 0
    assert R.paired_test(a, {99: 1.0})["n"] == 0                       # no shared seed
    assert R.paired_test({1: 2.0}, {1: 1.0})["p_two_sided"] is None    # one pair: no test
    same = R.paired_test({1: 2.0, 2: 3.0}, {1: 1.0, 2: 2.0})           # constant difference: sd = 0
    assert same["p_two_sided"] == 0.0 and same["t"] == math.inf
    zero = R.paired_test({1: 1.0, 2: 2.0}, {1: 1.0, 2: 2.0})
    assert zero["p_two_sided"] == 1.0


@pytest.fixture()
def replicate_configs(tmp_path):
    paths = []
    for seed, shift in ((42, 0.0), (43, 0.3), (44, 0.6)):
        cfg_file = tmp_path / f"seed{seed}.yaml"
        cfg_file.write_text(
            f"extends: {ROOT / 'configs' / 'quick.yaml'}\nseed: {seed}\n"
            f"paths: {{recipes: {tmp_path / f'rec{seed}'}, adapters: {tmp_path / f'ad{seed}'}, results: {tmp_path / f'res{seed}'}, cache: {tmp_path / 'cache'}}}\n")
        cfg = C.load_config(cfg_file)
        for cond in C.expand_conditions(cfg):
            if "ppl" in cond.evals:
                lora = 1.0 if cond.lora_rank else 0.0
                write_json(C.eval_path(cfg, cond, "ppl"), {"wikitext2_ppl": 20.0 - 4 * lora + shift})
        paths.append(str(cfg_file))
    return paths


def test_collect_and_report(replicate_configs, tmp_path):
    order, values, seeds = R.collect(replicate_configs)
    assert seeds == [42, 43, 44] and order[0][1] == "tiny-base"
    assert values[("tiny-base", "ppl")] == {42: 20.0, 43: 20.3, 44: 20.6}
    out = tmp_path / "rep"
    files = R.write_replicate_report(replicate_configs, out, compare=["tiny-heads-30", "tiny-heads-30-lora"], metric="ppl")
    assert {f.name for f in files} == {"replicates.csv", "REPLICATES.md", "paired.csv"}
    rows = (out / "replicates.csv").read_text().splitlines()
    assert rows[0].startswith("no,condition,metric,n,mean,sd")
    base = next(r for r in rows if r.split(",")[1] == "tiny-base" and r.split(",")[2] == "ppl")
    assert base.split(",")[3:6] == ["3", "20.3000", "0.3000"] and base.endswith("42 43 44")
    paired = (out / "paired.csv").read_text().splitlines()[1].split(",")
    assert paired[:3] == ["tiny-heads-30", "tiny-heads-30-lora", "ppl"]
    assert float(paired[4]) == pytest.approx(4.0)  # column "mean_diff": PPL without LoRA - PPL with LoRA


def test_replicates_must_differ(replicate_configs, tmp_path):
    with pytest.raises(ValueError, match="different seeds"):
        R.collect([replicate_configs[0], replicate_configs[0]])
    clash = tmp_path / "clash.yaml"
    clash.write_text(f"extends: {replicate_configs[0]}\nseed: 99\n")  # same output folders as seed 42
    with pytest.raises(ValueError, match="different results folders"):
        R.collect([replicate_configs[0], str(clash)])


def test_cli_aggregate(replicate_configs, tmp_path, capsys):
    out = tmp_path / "cli"
    assert main(["aggregate", "--configs", *replicate_configs, "--out", str(out)]) == 0
    assert (out / "REPLICATES.md").is_file() and "wrote 2 files" in capsys.readouterr().out


def test_shipped_replicate_configs_are_valid_and_isolated():
    cfgs = [C.load_config(ROOT / "configs" / n) for n in ("full.yaml", "replicates/seed43.yaml", "replicates/seed44.yaml")]
    assert [c["seed"] for c in cfgs] == [42, 43, 44]
    assert len({c["paths"]["results"] for c in cfgs}) == 3 and len({c["paths"]["adapters"] for c in cfgs}) == 3
    assert [len(C.expand_conditions(c)) for c in cfgs] == [len(C.expand_conditions(cfgs[0]))] * 3
