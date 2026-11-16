"""The analytical formulas must reproduce the thesis numbers (Tables 14, 21, 22, 24)."""
import pytest

from llmprune import counting as C

A7, A13 = C.LLAMA2_7B, C.LLAMA2_13B


def test_exact_huggingface_parameter_counts():
    assert C.total_params(A7) == 6_738_415_616
    assert C.total_params(A13) == 13_015_864_320


def test_head_pruning_30pct_7b():
    kept = 1024 - 307
    hpl = C.heads_per_layer_from_total(A7, kept)
    assert sum(hpl) == 717 and len(hpl) == 32
    assert C.attention_params(A7, 1) == 4 * 4096 * 128 == 2_097_152  # Equation 15
    removed = 307 * 2_097_152
    assert removed == pytest.approx(0.644e9, rel=1e-3)
    assert C.total_params(A7, hpl) == C.total_params(A7) - removed
    assert C.total_params(A7, hpl) / 1e9 == pytest.approx(6.095, abs=6e-4)
    assert C.reduction_pct(C.total_params(A7), C.total_params(A7, hpl)) == pytest.approx(9.6, abs=0.05)
    assert 32 * C.attention_params(A7, 32) / 1e9 == pytest.approx(2.147, abs=6e-4)
    assert 32 * C.ffn_params(A7) / 1e9 == pytest.approx(4.329, abs=6e-4)


def test_distribution_of_heads_does_not_change_totals():
    even = C.heads_per_layer_from_total(A7, 717)
    skew = [32] * 16 + [1] * 15 + [717 - 16 * 32 - 15]
    assert sum(skew) == 717
    assert C.total_params(A7, even) == C.total_params(A7, skew)
    assert C.flops_per_token(A7, even) == C.flops_per_token(A7, skew)


def test_layer_pruning_30pct_7b():
    n = 32 - 9
    assert C.total_params(A7, n_layers=n) / 1e9 == pytest.approx(4.917, abs=6e-4)
    assert C.reduction_pct(C.total_params(A7), C.total_params(A7, n_layers=n)) == pytest.approx(27.0, abs=0.05)
    assert C.compression_ratio(C.total_params(A7), C.total_params(A7, n_layers=n)) == pytest.approx(1.37, abs=0.005)


def test_magnitude_30pct_7b():
    prunable = C.prunable_params(A7)
    assert prunable / 1e9 == pytest.approx(6.476, abs=6e-4)
    zeroed = int(0.30 * prunable)
    assert zeroed / 1e9 == pytest.approx(1.943, abs=6e-4)
    nz = C.total_params(A7) - zeroed
    assert nz / 1e9 == pytest.approx(4.796, abs=6e-4)
    assert C.reduction_pct(C.total_params(A7), nz) == pytest.approx(28.8, abs=0.05)
    # The thesis prints 1.40 because it divides the rounded values (6.738 / 4.796 = 1.405 -> 1.40);
    # from exact counts the ratio is 1.4051, which rounds to 1.41.
    assert C.compression_ratio(C.total_params(A7), nz) == pytest.approx(1.405, abs=0.001)


@pytest.mark.parametrize("arch,kept,flops,mem0,mem1,params", [
    (A7, 1024 - 307, 11.3, 13.48, 12.19, 6.095),
    (A13, 1600 - 480, 11.0, 26.03, 23.52, 11.758),
])
def test_table22_head_rows(arch, kept, flops, mem0, mem1, params):
    hpl = C.heads_per_layer_from_total(arch, kept)
    prof = C.profile(arch, hpl)
    assert prof["flops_reduction_pct"] == pytest.approx(flops, abs=0.05)
    assert C.weight_gb(C.total_params(arch)) == pytest.approx(mem0, abs=0.006)
    assert prof["weight_gb_bf16"] == pytest.approx(mem1, abs=0.006)
    assert prof["structural_params"] / 1e9 == pytest.approx(params, abs=6e-4)


def test_table22_layer_row():
    prof = C.profile(A7, n_layers=23)
    assert prof["flops_reduction_pct"] == pytest.approx(27.6, abs=0.05)
    assert prof["weight_gb_bf16"] == pytest.approx(9.83, abs=0.006)


def test_kv_cache_per_token():
    assert C.kv_cache_bytes_per_token(A7) == 512 * 1024
    hpl = C.heads_per_layer_from_total(A7, 717)
    assert C.kv_cache_bytes_per_token(A7, hpl) / 1024 == pytest.approx(358.5, abs=0.01)


def test_lora_trainable_params_match_table24():
    hpl = C.heads_per_layer_from_total(A7, 512)  # 50 % of 1,024 heads
    expected_millions = {4: 3.15, 8: 6.29, 16: 12.58, 32: 25.17, 64: 50.33}
    for r, m in expected_millions.items():
        assert C.lora_trainable_params(A7, r, hpl) / 1e6 == pytest.approx(m, abs=0.005)


def test_gqa_is_rejected():
    gqa = C.Arch(hidden=64, ffn=128, layers=2, heads=8, vocab=100, kv_heads=2)
    with pytest.raises(NotImplementedError):
        C.total_params(gqa)
