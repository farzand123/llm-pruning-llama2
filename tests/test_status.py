"""Resume / stale-output logic (pure Python)."""
from pathlib import Path

import pytest

from llmprune import config as C
from llmprune import status as S
from llmprune.io_utils import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg(tmp_path):
    cfg = C.load_config(ROOT / "configs" / "quick.yaml")
    for key in ("recipes", "adapters", "results"):
        cfg["paths"][key] = str(tmp_path / key)
    return cfg


def cond(cfg, cid):
    return next(c for c in C.expand_conditions(cfg) if c.id == cid)


def finish(cfg, c, adapter_time="T1"):
    """Write plausible outputs for every stage of a condition."""
    if c.recipe_key:
        write_json(C.recipe_path(cfg, c), {"method": c.method})
    sha = S.recipe_sha(cfg, c)
    if c.lora_rank:
        write_json(C.adapter_dir(cfg, c) / "train_summary.json", {"finished": adapter_time, "provenance": {"recipe_sha256": sha}})
    prov = {"recipe_sha256": sha, "adapter_finished": adapter_time if c.lora_rank else None}
    for k in c.evals:
        write_json(C.eval_path(cfg, c, k), {"provenance": prov})
    if c.efficiency:
        for s in cfg["efficiency"]["settings"]:
            write_json(C.efficiency_path(cfg, c, s), {"provenance": prov})


def test_fresh_tree_needs_everything(cfg):
    plan = S.plan_actions(cfg, cond(cfg, "tiny-heads-30-lora"))
    assert plan["run"] == {"recipe": True, "lora": True, "eval": True, "efficiency": False}
    assert plan["eval_kinds"] == ["ppl", "glue"]


def test_baseline_has_no_recipe_or_lora(cfg):
    plan = S.plan_actions(cfg, cond(cfg, "tiny-base"))
    assert plan["run"] == {"recipe": False, "lora": False, "eval": True, "efficiency": True}
    assert plan["efficiency_settings"] == ["single"]


def test_complete_condition_is_skipped(cfg):
    c = cond(cfg, "tiny-heads-30")
    finish(cfg, c)
    assert S.is_complete(cfg, c)
    assert not any(S.plan_actions(cfg, c)["run"].values())


def test_missing_eval_runs_only_that_kind(cfg):
    c = cond(cfg, "tiny-heads-30")
    finish(cfg, c)
    C.eval_path(cfg, c, "glue").unlink()
    plan = S.plan_actions(cfg, c)
    assert plan["run"] == {"recipe": False, "lora": False, "eval": True, "efficiency": False}
    assert plan["eval_kinds"] == ["glue"]


def test_recomputed_recipe_makes_everything_downstream_stale(cfg):
    c = cond(cfg, "tiny-heads-30-lora")
    finish(cfg, c)
    write_json(C.recipe_path(cfg, c), {"method": "heads", "different": True})
    st = S.condition_status(cfg, c)
    assert len(st["lora"]["stale"]) == 1 and len(st["eval"]["stale"]) == len(c.evals)
    plan = S.plan_actions(cfg, c)
    assert plan["run"]["lora"] and plan["run"]["eval"] and not plan["run"]["recipe"]


def test_retrained_adapter_makes_evals_stale_but_not_the_adapter(cfg):
    c = cond(cfg, "tiny-heads-30-lora")
    finish(cfg, c)
    write_json(C.adapter_dir(cfg, c) / "train_summary.json", {"finished": "T2", "provenance": {"recipe_sha256": S.recipe_sha(cfg, c)}})
    st = S.condition_status(cfg, c)
    assert not st["lora"]["stale"] and len(st["eval"]["stale"]) == len(c.evals)
    plan = S.plan_actions(cfg, c)
    assert plan["run"] == {"recipe": False, "lora": False, "eval": True, "efficiency": False}


def test_truncated_result_file_counts_as_stale(cfg):
    c = cond(cfg, "tiny-heads-30")
    finish(cfg, c)
    C.eval_path(cfg, c, "ppl").write_text("{ not json")
    assert C.eval_path(cfg, c, "ppl") in S.condition_status(cfg, c)["eval"]["stale"]


def test_stage_selection_pulls_in_required_upstream_stages(cfg):
    plan = S.plan_actions(cfg, cond(cfg, "tiny-heads-30-lora"), stages=["eval"])
    assert plan["run"]["eval"] and plan["run"]["lora"] and plan["run"]["recipe"]  # eval needs an adapter, which needs a recipe


def test_stage_selection_does_not_run_unselected_work(cfg):
    plan = S.plan_actions(cfg, cond(cfg, "tiny-heads-30"), stages=["recipe"])
    assert plan["run"] == {"recipe": True, "lora": False, "eval": False, "efficiency": False}


def test_force_applies_to_selected_stage_and_downstream_only(cfg):
    c = cond(cfg, "tiny-heads-30-lora")
    finish(cfg, c)
    only_eval = S.plan_actions(cfg, c, stages=["eval"], force=True)
    assert only_eval["run"] == {"recipe": False, "lora": False, "eval": True, "efficiency": False}
    lora = S.plan_actions(cfg, c, stages=["lora"], force=True)
    assert lora["run"]["lora"] and not lora["run"]["recipe"]
