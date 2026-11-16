"""Parsing of lm-evaluation-harness results (pure Python, fabricated result dictionaries)."""
import pytest

from llmprune import metrics as M


def mmlu_results(acc=lambda i: 0.30 + 0.001 * i, key="mmlu_{}"):
    return {key.format(s): {"alias": s, "acc,none": acc(i), "acc_stderr,none": 0.01} for i, s in enumerate(M.MMLU_SUBJECTS)}


def test_get_metric_prefers_filtered_key_and_ignores_missing():
    assert M.get_metric({"acc,none": 0.5, "acc": 0.1}) == 0.5
    assert M.get_metric({"acc": 0.1}) == 0.1
    assert M.get_metric({"acc_stderr,none": 0.1}) is None
    assert M.get_metric(None) is None


def test_mmlu_is_a_macro_average_of_57_subjects():
    res = mmlu_results()
    res["mmlu"] = {"acc,none": 0.9}  # the harness' group value must not leak into the macro average
    out = M.summarise_mmlu(res, {"mmlu": {"acc,none": 0.4}})
    assert out["n_subjects"] == 57
    assert out["macro_average_pct"] == pytest.approx(100 * (0.30 + 0.001 * 28))
    assert out["harness_group_acc_pct"] == pytest.approx(40.0)  # groups take precedence over results
    assert out["per_subject_acc_pct"]["anatomy"] == pytest.approx(100 * 0.301)


def test_mmlu_old_task_names_are_understood():
    out = M.summarise_mmlu(mmlu_results(key="hendrycksTest-{}"))
    assert out["n_subjects"] == 57 and out["harness_group_acc_pct"] is None


def test_mmlu_missing_subjects_are_an_error_not_a_silent_average():
    res = mmlu_results()
    del res["mmlu_virology"]
    with pytest.raises(ValueError, match="virology"):
        M.summarise_mmlu(res)


def test_glue_average_is_the_unweighted_mean_of_the_tasks():
    res = {"sst2": {"acc,none": 0.946}, "mnli": {"acc,none": 0.824}, "qqp": {"acc,none": 0.879, "f1,none": 0.7},
           "qnli": {"acc,none": 0.913}}
    out = M.summarise_glue(res, ["sst2", "mnli", "qqp", "qnli"])
    assert out["acc_pct"]["qqp"] == pytest.approx(87.9)
    assert out["average_pct"] == pytest.approx((94.6 + 82.4 + 87.9 + 91.3) / 4)


def test_glue_missing_task_is_an_error():
    with pytest.raises(ValueError, match="qnli"):
        M.summarise_glue({"sst2": {"acc,none": 0.9}}, ["sst2", "qnli"])


def test_summaries_have_the_keys_the_report_reads():
    g = M.summarise_glue({t: {"acc,none": 0.5} for t in ("sst2", "mnli", "qqp", "qnli")}, ["sst2", "mnli", "qqp", "qnli"])
    assert set(g) == {"acc_pct", "average_pct"}
    m = M.summarise_mmlu(mmlu_results())
    assert "macro_average_pct" in m
