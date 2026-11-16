from pathlib import Path

import pytest

from llmprune import config as K

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def full():
    return K.load_config(ROOT / "configs" / "full.yaml")


def test_matrix_matches_thesis_table17(full):
    conds = K.expand_conditions(full)
    by = {c.id: c for c in conds}
    assert len(conds) == 46 and len(by) == 46
    assert sorted({c.no for c in conds}) == list(range(42))     # rows 0..41 of Table 17
    assert by["7b-magnitude-10"].no == 1 and by["7b-magnitude-70-lora"].no == 12
    assert by["7b-heads-10"].no == 13 and by["7b-heads-70-lora"].no == 24
    assert by["7b-layers-10"].no == 25 and by["7b-layers-70-lora"].no == 36
    assert by["13b-heads-10"].no == 37 and by["13b-heads-30-lora"].no == 40
    assert {c.no for c in conds if c.group == "rank-ablation"} == {41}


def test_reported_results_per_condition(full):
    by = {c.id: c for c in K.expand_conditions(full)}
    assert by["7b-base"].evals == ("ppl", "glue", "mmlu") and by["7b-base"].efficiency
    assert by["13b-base"].evals == () and by["13b-base"].efficiency
    assert by["7b-heads-30"].evals == ("ppl", "glue", "mmlu") and by["7b-heads-30"].efficiency
    assert by["7b-heads-50"].evals == ("ppl", "mmlu") and not by["7b-heads-50"].efficiency
    assert by["7b-heads-30-lora"].evals == ("ppl", "glue") and by["7b-heads-30-lora"].lora_rank == 16
    assert by["7b-heads-50-lora"].evals == ("ppl", "mmlu")
    assert by["7b-layers-70-lora"].evals == ("ppl",)
    assert by["13b-heads-30"].efficiency and by["13b-heads-10"].is_noop
    assert not by["13b-heads-30-lora"].enabled
    assert {by[f"7b-heads-50-lora-r{r}"].lora_rank for r in (4, 8, 32, 64)} == {4, 8, 32, 64}
    assert "7b-heads-50-lora-r16" not in by  # r = 16 is the main LoRA condition


def test_recipe_is_shared_between_plain_and_lora(full):
    by = {c.id: c for c in K.expand_conditions(full)}
    assert by["7b-heads-30"].recipe_key == by["7b-heads-30-lora"].recipe_key == "7b-heads-30"
    assert by["7b-base"].recipe_key is None


def test_selection_and_sharding(full):
    conds = K.expand_conditions(full)
    assert [c.id for c in K.select_conditions(conds, only=["7b-heads-3*"])] == ["7b-heads-30", "7b-heads-30-lora"]
    assert [c.id for c in K.select_conditions(conds, only=["15"])] == ["7b-heads-30"]
    assert all(c.enabled for c in K.select_conditions(conds))
    shards = [K.select_conditions(conds, shard=(k, 3)) for k in range(3)]
    assert sorted(c.id for s in shards for c in s) == sorted(c.id for c in conds if c.enabled)
    with pytest.raises(ValueError):
        K.select_conditions(conds, shard=(3, 3))


def test_quick_config_inherits_and_validates():
    q = K.load_config(ROOT / "configs" / "quick.yaml")
    assert q["lora"]["rank"] == 8 and q["lora"]["dropout"] == 0.05          # overridden / inherited
    assert q["matrix"]["secondary_model"] is None
    conds = K.expand_conditions(q)
    assert all(c.model_key == "tiny" for c in conds)
    assert len({c.id for c in conds}) == len(conds)


def test_validation_errors(full):
    bad = K.deep_merge(full, {"matrix": {"ratios": [0.3, 1.2]}})
    with pytest.raises(ValueError):
        K.validate(bad)
    bad = K.deep_merge(full, {"runtime": {"dtype": "float32"}})
    with pytest.raises(ValueError):
        K.validate(bad)
    bad = K.deep_merge(full, {"matrix": {"methods": ["magnitude", "wanda"]}})
    with pytest.raises(ValueError):
        K.validate(bad)


def test_expected_outputs_are_empty_for_noops(full):
    by = {c.id: c for c in K.expand_conditions(full)}
    assert K.expected_outputs(full, by["13b-heads-10"]) == []
    outs = K.expected_outputs(full, by["7b-heads-30-lora"])
    assert any(str(p).endswith("train_summary.json") for p in outs)
    assert any(p.name == "eval_glue.json" for p in outs)
