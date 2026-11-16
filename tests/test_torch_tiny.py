"""PyTorch half of the repository, on a tiny random LLaMA (CPU, no downloads).

Skipped when torch / transformers are not installed. The same checks run with ``llmprune selftest``.
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from llmprune import selftest as S  # noqa: E402


@pytest.mark.torch
@pytest.mark.parametrize("name,check", S.CHECKS, ids=[n for n, _ in S.CHECKS])
def test_selftest_check(name, check):
    try:
        detail = check()
    except ImportError as exc:  # e.g. peft not installed
        pytest.skip(f"optional dependency missing: {exc}")
    assert detail


@pytest.mark.torch
def test_head_verification_tolerances(caplog):
    from llmprune.pruning import check_head_verification

    check_head_verification({"rel_l2": 1e-6, "max_abs": 1e-5, "argmax_agreement": 1.0}, "float32")
    check_head_verification({"rel_l2": 0.05, "max_abs": 0.5, "argmax_agreement": 0.95}, "bfloat16")
    with pytest.raises(RuntimeError):  # gross error: a wrong index gives a relative error of order one
        check_head_verification({"rel_l2": 0.9, "max_abs": 9.0, "argmax_agreement": 0.1}, "bfloat16")
    with pytest.raises(RuntimeError):  # float32 must match to rounding error
        check_head_verification({"rel_l2": 1e-2, "max_abs": 1e-1, "argmax_agreement": 1.0}, "float32")
    with caplog.at_level("WARNING"):  # low argmax agreement alone is only reported in half precision
        check_head_verification({"rel_l2": 0.05, "max_abs": 0.5, "argmax_agreement": 0.4}, "bfloat16")
    assert "argmax agreement" in caplog.text
