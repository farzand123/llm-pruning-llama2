import math

import pytest

from llmprune import metrics as M


def test_recovery_ratio_perplexity_and_accuracy():
    # Equation 14 on thesis values: magnitude, 30 % (PPL) and 50 % (MMLU)
    assert M.recovery_ratio(6.97, 7.61, 7.31) == pytest.approx(0.469, abs=1e-3)
    assert M.recovery_ratio(45.3, 39.8, 42.3, higher_is_better=True) == pytest.approx(0.4545, abs=1e-3)
    assert M.recovery_ratio(10.0, 20.0, 10.0) == 1.0 and M.recovery_ratio(10.0, 20.0, 20.0) == 0.0
    assert math.isnan(M.recovery_ratio(7.0, 7.0, 7.0))


def test_choose_rank_reproduces_thesis_table24():
    ppl = {4: 9.84, 8: 8.91, 16: 8.14, 32: 7.98, 64: 7.96}
    mmlu = {4: 41.6, 8: 42.7, 16: 43.8, 32: 43.9, 64: 43.9}
    r, table = M.choose_rank(ppl, mmlu)
    assert r == 16
    assert [row["satisfies"] for row in table] == [False, False, True, True]
    assert table[2]["d_ppl"] == pytest.approx(0.16) and table[2]["d_mmlu"] == pytest.approx(0.1)


def test_choose_rank_none_when_criterion_never_met():
    assert M.choose_rank({4: 10, 8: 8}, {4: 40, 8: 42})[0] is None
    assert M.choose_rank({4: 10}, {4: 40})[0] is None


def test_mmlu_subject_list():
    assert len(M.MMLU_SUBJECTS) == 57 == len(set(M.MMLU_SUBJECTS))


def test_formatting_helpers():
    assert M.fmt(None) == "–" and M.fmt(float("nan")) == "–"
    assert M.fmt(1.234, 1, "%") == "1.2%"
    assert M.macro_average([1, 2, 3]) == 2
    assert M.speedup(56.7, 41.2) == pytest.approx(1.376, abs=1e-3)
