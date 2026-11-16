#!/usr/bin/env python
"""Reproduce the thesis' analytical numbers (no GPU, no model download, no PyTorch).

Parameter counts, pruned-head counts, FLOPs (Equation 13), KV-cache size, LoRA trainable parameters
(Table 24) and the compression ratios of Table 21 follow from the LLaMA-2 architecture alone. Every line
below is compared with the value printed in the thesis; the script exits with status 1 on a mismatch.

    python scripts/analytic_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmprune import counting as C  # noqa: E402
from llmprune import selection as sel  # noqa: E402
from llmprune.report import thesis_reported  # noqa: E402

T = thesis_reported()
failures = 0


def check(label, value, expected, tol=0.0, fmt="{:,.3f}"):
    global failures
    ok = abs(value - expected) <= tol
    failures += not ok
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label:<58} {fmt.format(value):>18}   thesis: {fmt.format(expected)}")


a7, a13 = C.LLAMA2_7B, C.LLAMA2_13B
print("Parameter counts (structural)")
check("LLaMA-2-7B total parameters", C.total_params(a7), 6_738_415_616, fmt="{:,.0f}")
check("LLaMA-2-13B total parameters", C.total_params(a13), 13_015_864_320, fmt="{:,.0f}")
check("7B attention-projection parameters (billions)", 32 * C.attention_params(a7, 32) / 1e9, 2.147, 5e-4)
check("7B FFN parameters (billions)", 32 * C.ffn_params(a7) / 1e9, 4.329, 5e-4)
check("7B prunable weights (billions)", C.prunable_params(a7) / 1e9, 6.476, 5e-4)
check("Parameters of one head, Equation 15", C.attention_params(a7, 1), 2_097_152, fmt="{:,.0f}")

print("\nPruned-component counts (floor(r * total))")
for p, heads, layers in ((10, 102, 3), (20, 204, 6), (30, 307, 9), (40, 409, 12), (50, 512, 16), (70, 716, 22)):
    check(f"{p}% -> heads removed of 1,024", sel.n_to_remove(p / 100, 1024), heads, fmt="{:,.0f}")
    check(f"{p}% -> layers removed of 32", sel.n_to_remove(p / 100, 32), layers, fmt="{:,.0f}")

print("\n30 % pruning of LLaMA-2-7B (Tables 21 and 22)")
base_total = C.total_params(a7)
zeroed = sel.n_to_remove(0.30, C.prunable_params(a7))
check("magnitude: weights set to zero (billions)", zeroed / 1e9, 1.943, 5e-4)
check("magnitude: non-zero parameters (billions)", (base_total - zeroed) / 1e9, T["params_30"]["magnitude"]["nonzero_b"], 5e-4)
check("magnitude: non-zero reduction (%)", C.reduction_pct(base_total, base_total - zeroed), 28.8, 0.05, "{:.2f}")
check("magnitude: CR_nz (thesis divides rounded values: 1.40)", C.compression_ratio(base_total, base_total - zeroed), 1.40, 0.006, "{:.3f}")
kept = 1024 - 307
hpl = C.heads_per_layer_from_total(a7, kept)
check("heads: total parameters (billions)", C.total_params(a7, hpl) / 1e9, T["params_30"]["heads"]["structural_b"], 5e-4)
check("heads: parameter reduction (%)", C.reduction_pct(base_total, C.total_params(a7, hpl)), 9.6, 0.06, "{:.2f}")
check("heads: KV cache per token (KiB, bfloat16)", C.kv_cache_bytes_per_token(a7, hpl) / 1024, 358.5, 0.06, "{:.1f}")
check("baseline KV cache per token (KiB, bfloat16)", C.kv_cache_bytes_per_token(a7) / 1024, 512.0, 0.0, "{:.1f}")
check("heads: FLOPs reduction, Equation 13 (%)", C.profile(a7, hpl)["flops_reduction_pct"], T["efficiency_30"]["7b-heads"]["flops_reduction_pct"], 0.06, "{:.2f}")
check("layers: total parameters (billions)", C.total_params(a7, n_layers=23) / 1e9, T["params_30"]["layers"]["structural_b"], 5e-4)
check("layers: FLOPs reduction, Equation 13 (%)", C.profile(a7, n_layers=23)["flops_reduction_pct"], T["efficiency_30"]["7b-layers"]["flops_reduction_pct"], 0.06, "{:.2f}")
check("layers: realised fraction of blocks removed (%)", 100 * 9 / 32, 28.1, 0.05, "{:.2f}")

print("\nLoRA trainable parameters, 50 % head pruning (Table 24, millions)")
hpl50 = C.heads_per_layer_from_total(a7, 1024 - 512)
for rank, row in T["rank_ablation"].items():
    check(f"rank {rank}", C.lora_trainable_params(a7, int(rank), hpl50) / 1e6, row["trainable_millions"], 0.006, "{:.2f}")

print(f"\n{'all analytical checks match the thesis' if not failures else str(failures) + ' MISMATCH(ES): see above'}")
sys.exit(1 if failures else 0)
