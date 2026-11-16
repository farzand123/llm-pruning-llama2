"""Command-line interface on the pure-Python path (no PyTorch needed)."""
from pathlib import Path

import pytest

from llmprune.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg_path(tmp_path):
    p = tmp_path / "test.yaml"
    p.write_text(
        f"extends: {ROOT / 'configs' / 'quick.yaml'}\n"
        f"paths: {{recipes: {tmp_path / 'recipes'}, adapters: {tmp_path / 'adapters'}, results: {tmp_path / 'results'}, cache: {tmp_path / 'cache'}}}\n"
    )
    return str(p)


def test_plan_prints_matrix_and_totals(cfg_path, capsys):
    assert main(["plan", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "tiny-heads-30-lora" in out and "LoRA trainings" in out


def test_plan_of_the_thesis_matrix(capsys):
    assert main(["plan", "--config", str(ROOT / "configs" / "full.yaml")]) == 0
    out = capsys.readouterr().out
    assert "7b-layers-70-lora" in out and "13b-heads-30" in out
    assert "18 pruning recipes" not in out  # 3 methods x 6 ratios + 3 cross-scale ones
    assert "21 pruning recipes" in out


def test_dry_run_lists_stages(cfg_path, capsys):
    assert main(["run", "--config", cfg_path, "--only", "tiny-base", "tiny-heads-30-lora", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "tiny-base" in out and "-> eval, efficiency" in out
    assert "-> recipe, lora, eval" in out


def test_selection_by_number_glob_and_shard(cfg_path, capsys):
    assert main(["plan", "--config", cfg_path, "--only", "tiny-magnitude-*"]) == 0
    out = capsys.readouterr().out
    assert "tiny-magnitude-30" in out and "tiny-heads-30" not in out
    assert main(["plan", "--config", cfg_path, "--shard", "0/2"]) == 0
    assert main(["plan", "--config", cfg_path, "--only", "5"]) == 0  # Table 17 number


def test_unmatched_selection_is_an_error(cfg_path):
    with pytest.raises(SystemExit):
        main(["plan", "--config", cfg_path, "--only", "does-not-exist"])


def test_status_on_empty_tree(cfg_path, capsys):
    assert main(["status", "--config", cfg_path]) == 0
    out = capsys.readouterr().out
    assert "recipe 0/1" in out and "0 of 15 conditions complete" in out


def test_report_runs_on_an_empty_tree(cfg_path, tmp_path, capsys):
    out_dir = tmp_path / "rep"
    assert main(["report", "--config", cfg_path, "--out", str(out_dir)]) == 0
    assert (out_dir / "REPORT.md").is_file() and (out_dir / "table18_ppl.csv").is_file()
    assert (out_dir / "comparison_with_thesis.csv").is_file()


def test_unknown_stage_is_rejected(cfg_path):
    with pytest.raises(SystemExit):
        main(["run", "--config", cfg_path, "--stages", "train", "--dry-run"])
