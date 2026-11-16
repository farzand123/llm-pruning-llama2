"""Self-test on a tiny random LLaMA (no downloads, no GPU, a few seconds).

Run it with ``llmprune selftest`` (or ``pytest -m torch``) before starting the multi-day matrix. It checks, on a
4-layer model with 4 heads per layer, the properties the thesis results depend on:

* physical head removal gives exactly the logits of "the same model with those heads masked out",
* deleting layers equals skipping them, and the KV cache still works afterwards (layer renumbering),
* the exact magnitude threshold zeroes what the stored recipe says it zeroes,
* the Taylor head scores equal the gradient with respect to an explicit gate variable (Michel et al.),
* Block Influence is exactly zero for a block that is an identity map,
* recipes survive a JSON round trip and replay to a bit-identical model,
* PEFT's trainable-parameter count equals the thesis formula (Table 24), LoRA learns, the base stays frozen,
* the sliding-window perplexity matches a brute-force reference,
* the analytical parameter counts of ``counting.py`` equal what PyTorch counts.
"""
from __future__ import annotations

import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Tuple

import numpy as np
import torch

from . import counting as C
from . import selection as sel
from .config import Condition
from .importance import block_influence, taylor_head_importance
from .io_utils import read_json, write_json
from .modeling import (apply_head_pruning, apply_pattern_threshold, compare_logits, count_parameters, describe_shapes,
                       fingerprint, logits_with_heads_masked, magnitude_prune, prunable_linears, remove_layers,
                       require_mha, unwrap)

VOCAB, HIDDEN, HEADS, FFN, LAYERS = 128, 64, 4, 96, 4


def tiny_llama(dtype=torch.float32, layers: int = LAYERS, seed: int = 0, attn: str = None, kv_heads: int = None):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=FFN, num_hidden_layers=layers,
                      num_attention_heads=HEADS, num_key_value_heads=kv_heads or HEADS, max_position_embeddings=256,
                      rms_norm_eps=1e-5, tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    if attn:
        cfg._attn_implementation = attn
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).to(dtype).eval()


def _ids(n: int, t: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(3, VOCAB, (n, t), generator=torch.Generator().manual_seed(seed))


def _arch(model) -> C.Arch:
    return C.arch_from_config(unwrap(model).config)


def _assert_close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> float:
    rel = float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12))
    assert rel < tol, f"{what}: relative L2 difference {rel:.3e} exceeds {tol:g}"
    return rel


def _step_vs_full_forward(model, ids: torch.Tensor, tol: float = 1e-4) -> float:
    """One cached decoding step must reproduce the last logits of a full forward pass."""
    with torch.no_grad():
        full = model(ids, use_cache=False).logits[:, -1]
        prefill = model(ids[:, :-1], use_cache=True)
        step = model(ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True).logits[:, -1]
    err = float((full - step).abs().max())
    assert err < tol, f"cached decoding differs from a full forward pass by {err:.3e}"
    return err


# --------------------------------------------------------------------------- checks
def check_parameter_counts() -> str:
    m = tiny_llama()
    assert count_parameters(m)["structural"] == C.total_params(_arch(m)), "analytic total_params != PyTorch count"
    keep = [[0, 1], [2], [0, 1, 3], [1, 2]]
    apply_head_pruning(m, keep)
    heads = [len(k) for k in keep]
    assert count_parameters(m)["structural"] == C.total_params(_arch(tiny_llama()), heads)
    kv = describe_shapes(m)["kv_bytes_per_token"]
    assert kv == C.kv_cache_bytes_per_token(_arch(tiny_llama()), heads, bytes_per_element=4), "KV-cache size mismatch"
    m2 = tiny_llama()
    remove_layers(m2, [1, 2])
    assert count_parameters(m2)["structural"] == C.total_params(_arch(tiny_llama()), n_layers=2)
    return "analytic parameter counts equal torch.numel for dense, head-pruned and layer-pruned models"


def check_head_pruning() -> str:
    errs = []
    for attn in ("eager", "sdpa"):
        m = tiny_llama(attn=attn)
        ids = _ids(2, 24)
        scores = np.random.default_rng(0).random((LAYERS, HEADS))
        mask, info = sel.select_heads_to_remove(scores, 0.4)
        with torch.no_grad():
            dense = m(ids, use_cache=False).logits
            ref = logits_with_heads_masked(m, mask, ids)
        apply_head_pruning(m, sel.keep_lists(mask))
        with torch.no_grad():
            got = m(ids, use_cache=False).logits
        errs.append(_assert_close(ref, got, 1e-5, f"pruned vs masked ({attn})"))
        assert float((dense - got).norm() / dense.norm()) > 1e-5, "pruning changed nothing (vacuous test)"
        shapes = describe_shapes(m)
        assert shapes["n_heads_total"] == info["n_kept"], (shapes["n_heads_total"], info["n_kept"])
        errs.append(_step_vs_full_forward(m, ids))
    return f"pruned == masked reference (eager and sdpa), cached decoding ok, max error {max(errs):.1e}"


def check_layer_removal() -> str:
    m = tiny_llama(layers=5)
    ids = _ids(2, 20)

    def skip(module, args, kwargs, output):
        xi = args[0] if args else kwargs["hidden_states"]
        return (xi,) + tuple(output[1:]) if isinstance(output, tuple) else xi

    handles = [m.model.layers[i].register_forward_hook(skip, with_kwargs=True) for i in (1, 3)]
    with torch.no_grad():
        ref = m(ids, use_cache=False).logits
    for h in handles:
        h.remove()
    remove_layers(m, [1, 3])
    with torch.no_grad():
        got = m(ids, use_cache=False).logits
    rel = _assert_close(ref, got, 1e-6, "layer deletion vs identity skip")
    assert [l.self_attn.layer_idx for l in m.model.layers] == [0, 1, 2], "layers were not renumbered"
    assert m.config.num_hidden_layers == 3
    err = _step_vs_full_forward(m, ids)  # would index past the end of the KV cache without renumbering
    return f"deleted == skipped (rel. error {rel:.1e}); KV cache consistent after renumbering ({err:.1e})"


def check_magnitude() -> str:
    m = tiny_llama(dtype=torch.bfloat16)
    original = [mod.weight.detach().clone() for _, mod in prunable_linears(m)]
    info = magnitude_prune(m, 0.3)
    assert abs(info["realised_ratio"] - 0.3) < 0.01, info
    pruned_max, kept_min = 0.0, float("inf")
    n_zeroed = 0
    for w0, (_, mod) in zip(original, prunable_linears(m)):
        w = mod.weight.detach()
        zeroed = (w == 0) & (w0 != 0)
        n_zeroed += int(zeroed.sum())
        if zeroed.any():
            pruned_max = max(pruned_max, float(w0[zeroed].float().abs().max()))
        if (w != 0).any():
            kept_min = min(kept_min, float(w[w != 0].float().abs().min()))
    assert pruned_max < kept_min, f"threshold is not monotone: pruned up to {pruned_max}, kept from {kept_min}"
    assert n_zeroed == info["zeroed"], (n_zeroed, info["zeroed"])
    fresh = tiny_llama(dtype=torch.bfloat16)
    assert apply_pattern_threshold(fresh, info["pattern_threshold"]) == info["zeroed"], "stored threshold does not replay"
    return f"{info['zeroed']} of {info['prunable_weights']} weights zeroed ({100 * info['realised_ratio']:.2f}% for a 30% target); threshold replays exactly"


def check_taylor_scores() -> str:
    m = tiny_llama()
    calib = _ids(3, 16)
    got = taylor_head_importance(m, calib, batch_size=1, show_progress=False)
    hd = HIDDEN // HEADS
    ref = np.zeros((LAYERS, HEADS))
    for p in m.parameters():
        p.requires_grad_(False)
    for row in calib:
        gates = [torch.ones(HEADS, requires_grad=True) for _ in range(LAYERS)]

        def make_pre(gate):
            def pre(module, args):
                x = args[0]
                b, t, _ = x.shape
                return ((x.reshape(b, t, HEADS, hd) * gate.view(1, 1, HEADS, 1)).reshape(b, t, HEADS * hd),)
            return pre

        handles = [layer.self_attn.o_proj.register_forward_pre_hook(make_pre(g)) for layer, g in zip(m.model.layers, gates)]
        m(row[None], labels=row[None], use_cache=False).loss.backward()
        for h in handles:
            h.remove()
        for i, g in enumerate(gates):
            ref[i] += g.grad.abs().numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-7)
    assert got.shape == (LAYERS, HEADS) and np.isfinite(got).all() and (got >= 0).all()
    return "Taylor scores equal |dL/d gate| of an explicit head gate (Michel et al.)"


def check_block_influence() -> str:
    m = tiny_llama()
    calib = _ids(4, 16)
    with torch.no_grad():
        layer = m.model.layers[2]
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()  # block 2 is now an identity map
    bi = block_influence(m, calib, batch_size=2, show_progress=False)
    assert abs(bi["frobenius"][2]) < 1e-6 and abs(bi["cosine"][2]) < 1e-6, bi
    assert min(v for i, v in enumerate(bi["frobenius"]) if i != 2) > 1e-4
    assert sel.select_layers_to_remove(bi["frobenius"], 0.25) == [2], "the identity block must be removed first"
    # independent per-sequence computation for block 0
    frob, cosd = [], []
    for row in calib:
        captured = {}

        def hook(module, args, kwargs, output):
            captured["i"] = (args[0] if args else kwargs["hidden_states"]).detach().float()
            captured["o"] = (output[0] if isinstance(output, tuple) else output).detach().float()

        h = m.model.layers[0].register_forward_hook(hook, with_kwargs=True)
        with torch.no_grad():
            m.model(input_ids=row[None], use_cache=False)
        h.remove()
        xi, xo = captured["i"], captured["o"]
        frob.append(float((xo - xi).norm() / xi.norm()))
        cosd.append(float((1 - torch.nn.functional.cosine_similarity(xi, xo, dim=-1)).mean()))
    assert abs(np.mean(frob) - bi["frobenius"][0]) < 1e-5 and abs(np.mean(cosd) - bi["cosine"][0]) < 1e-5
    return "BI = 0 for an identity block; both definitions equal an independent per-sequence computation"


def _cfg(dtype: str) -> dict:
    return {"seed": 0, "runtime": {"dtype": dtype}, "calibration": {"dataset": "random", "c4_shard": None, "batch_size": 2},
            "pruning": {"heads": {"min_heads_per_layer": 1, "verify": True, "taylor_batch_size": 1},
                        "layers": {"bi_variant": "frobenius"}}}


def check_recipe_roundtrip() -> str:
    from .pruning import apply_recipe, make_recipe

    calib = _ids(4, 32)
    spec = {"name_or_path": "random-tiny", "revision": None}
    done = []
    for method, pct, dtype in (("heads", 40, torch.float32), ("heads", 40, torch.bfloat16),
                               ("layers", 50, torch.float32), ("magnitude", 30, torch.bfloat16)):
        name = {torch.float32: "float32", torch.bfloat16: "bfloat16"}[dtype]
        cond = Condition(no=1, id=f"tiny-{method}-{pct}", model_key="tiny", method=method, pct=pct, lora_rank=None,
                         evals=(), efficiency=False)
        m1 = tiny_llama(dtype)
        recipe = make_recipe(m1, None if method == "magnitude" else calib, _cfg(name), cond, spec)
        with tempfile.TemporaryDirectory() as d:
            write_json(Path(d) / "r.json", recipe)
            stored = read_json(Path(d) / "r.json")
        m2 = apply_recipe(tiny_llama(dtype), stored)
        for (_, a), (_, b) in zip(m1.named_parameters(), m2.named_parameters()):
            assert a.shape == b.shape and torch.equal(a, b), "replayed recipe differs from the pruned model"
        assert count_parameters(m2) == stored["params_after"]
        if method == "heads":
            keep = stored["heads_keep"]
            assert sum(len(k) for k in keep) == stored["head_selection"]["n_kept"] == describe_shapes(m2)["n_heads_total"]
            assert stored["verification"]["rel_l2"] < 0.15
        if method == "layers":
            assert len(stored["layers_removed"]) == 2 and len(stored["bi_scores"]["frobenius"]) == LAYERS
        done.append(f"{method}/{name}")
    return "make_recipe -> JSON -> apply_recipe is bit-identical for " + ", ".join(done)


def check_lora() -> str:
    from torch.utils.data import DataLoader

    from .data import SupervisedDataset, make_collate
    from .lora import attach_lora, train

    base = tiny_llama(dtype=torch.bfloat16)
    keep = [[0, 1], [2], [0, 1, 3], [1, 2]]
    apply_head_pruning(base, keep)
    heads = [len(k) for k in keep]
    before = fingerprint(base)
    pm = attach_lora(base, rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainable = [p for p in pm.parameters() if p.requires_grad]
    assert all(p.dtype == torch.float32 for p in trainable), "adapters must be float32"
    n = sum(p.numel() for p in trainable)
    expected = C.lora_trainable_params(_arch(tiny_llama()), 4, heads)
    assert n == expected, f"PEFT trainable parameters {n} != thesis formula {expected}"
    seqs = _ids(4, 16)
    ds = SupervisedDataset([r.tolist() for r in seqs], [r.tolist() for r in seqs])
    loader = DataLoader(ds, batch_size=2, collate_fn=make_collate(0), drop_last=True)
    stats = train(pm, loader, lr=1e-2, weight_decay=0.0, epochs=6, grad_accum=2, warmup_steps=1, max_grad_norm=1.0)
    assert stats["steps"] == 6 and np.isfinite(stats["final_loss"]), stats
    b_norm = sum(float(p.detach().abs().sum()) for nme, p in pm.named_parameters() if "lora_B" in nme)
    assert b_norm > 0, "LoRA B matrices are still zero: no gradient reached the adapters"
    after = fingerprint(pm)
    assert after["nonzero_prunable"] == before["nonzero_prunable"] and abs(after["abs_sum_prunable"] - before["abs_sum_prunable"]) < 1e-6 * max(1.0, before["abs_sum_prunable"]), "base weights changed"
    with tempfile.TemporaryDirectory() as d:
        pm.save_pretrained(d)
        assert (Path(d) / "adapter_config.json").is_file()
    return f"{n} trainable parameters (= 4r*sum(d + h*head_dim)); loss {stats['first_loss']:.3f} -> {stats['final_loss']:.3f}; base frozen"


def check_perplexity() -> str:
    from .evaluation import sliding_window_nll

    m = tiny_llama()
    ids = _ids(1, 70)
    r = sliding_window_nll(m, ids, window=32, stride=16)
    nll, n_tok, prev_end = 0.0, 0, 0
    with torch.no_grad():
        for begin in range(0, 70, 16):
            end = min(begin + 32, 70)
            x = ids[:, begin:end]
            logp = torch.log_softmax(m(x, use_cache=False).logits[0].double(), dim=-1)
            for j in range(1, x.size(1)):
                if begin + j >= prev_end:
                    nll -= float(logp[j - 1, x[0, j]])
                    n_tok += 1
            prev_end = end
            if end == 70:
                break
    assert n_tok == r["n_tokens"] == 69, (n_tok, r["n_tokens"])
    assert abs(nll - r["nll_sum"]) < 1e-2, (nll, r["nll_sum"])
    return f"sliding-window NLL equals a brute-force reference over {n_tok} tokens (perplexity {r['ppl']:.1f}, ~{VOCAB} for a random model)"


def check_benchmark() -> str:
    from .benchmark import benchmark_generation

    m = tiny_llama()
    res = benchmark_generation(m, SimpleNamespace(pad_token_id=0), prompt_len=8, new_tokens=6, batch=1, runs=2, warmup=1)
    assert res["tokens_per_s_mean"] > 0 and res["protocol"]["new_tokens"] == 6
    return f"generation benchmark runs ({res['tokens_per_s_mean']:.0f} tokens/s on this machine)"


def check_gqa_rejected() -> str:
    m = tiny_llama(kv_heads=2)
    try:
        require_mha(m)
    except NotImplementedError:
        return "grouped-query attention is rejected with a clear error"
    raise AssertionError("GQA model was not rejected")


CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("parameter counts vs counting.py", check_parameter_counts),
    ("head pruning == masked heads", check_head_pruning),
    ("layer removal == identity skip", check_layer_removal),
    ("magnitude threshold", check_magnitude),
    ("Taylor head importance", check_taylor_scores),
    ("Block Influence", check_block_influence),
    ("recipe round trip", check_recipe_roundtrip),
    ("LoRA", check_lora),
    ("sliding-window perplexity", check_perplexity),
    ("efficiency benchmark", check_benchmark),
    ("GQA guard", check_gqa_rejected),
]


def run_all(verbose: bool = True) -> bool:
    ok = True
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            detail = fn()
            if verbose:
                print(f"  PASS  {name:<34} {detail}  ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001 - report and continue with the other checks
            ok = False
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print("selftest: all checks passed" if ok else "selftest: FAILED - do not start the long runs before this passes")
    return ok
