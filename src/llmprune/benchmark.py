"""Efficiency benchmark (PyTorch) - thesis Section III.F with the unstated details fixed.

Protocol: a deterministic random prompt of ``prompt_len`` tokens, greedy decoding of exactly
``new_tokens`` new tokens (``min_new_tokens`` prevents early EOS), batch size ``batch``, ``warmup``
untimed passes, then ``runs`` timed passes. Each pass is bracketed by ``cuda.synchronize`` on every
GPU that holds weights. Throughput = batch * new_tokens / wall-clock seconds of the whole
``generate`` call (so it includes the prefill of the prompt). Memory is ``torch.cuda.memory_allocated``
(tensors only, no allocator cache), in decimal GB.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional

import torch

from .modeling import count_parameters, embed_device, hardware_info, unwrap


def cuda_devices_used(model) -> List[int]:
    return sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})


def allocated_bytes(devices: Optional[List[int]] = None) -> Dict[int, int]:
    """memory_allocated of the given CUDA devices (all visible devices by default)."""
    if not torch.cuda.is_available():
        return {}
    ids = range(torch.cuda.device_count()) if devices is None else devices
    return {d: int(torch.cuda.memory_allocated(d)) for d in ids}


def _sync(devices: List[int]) -> None:
    for d in devices:
        torch.cuda.synchronize(d)


@torch.no_grad()
def benchmark_generation(model, tokenizer, *, prompt_len: int, new_tokens: int, batch: int, runs: int,
                         warmup: int, seed: int = 42, baseline_allocated: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
    """Measure generation throughput and GPU memory of an already loaded model.

    ``baseline_allocated`` = ``allocated_bytes()`` taken *before* the model was loaded; it is subtracted
    so that leftovers of earlier work do not count as weights.
    """
    base = unwrap(model)
    dev = embed_device(model)
    devices = cuda_devices_used(model)
    baseline_allocated = baseline_allocated or {}
    vocab = int(base.config.vocab_size)

    gen = torch.Generator().manual_seed(seed)
    prompt = torch.randint(low=min(5, vocab - 1), high=vocab, size=(batch, prompt_len), generator=gen).to(dev)
    attention_mask = torch.ones_like(prompt)
    gen_kwargs = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, temperature=None,
                      top_p=None, use_cache=True, pad_token_id=tokenizer.pad_token_id)

    now = allocated_bytes(devices)
    weights_bytes = sum(max(0, now.get(d, 0) - baseline_allocated.get(d, 0)) for d in devices)

    # input_ids is passed by keyword: PeftModel.generate of some PEFT versions accepts keyword arguments only
    for _ in range(warmup):
        model.generate(input_ids=prompt, attention_mask=attention_mask, **gen_kwargs)
    if devices:
        _sync(devices)
        for d in devices:
            torch.cuda.reset_peak_memory_stats(d)

    seconds: List[float] = []
    for _ in range(runs):
        if devices:
            _sync(devices)
        t0 = time.perf_counter()
        out = model.generate(input_ids=prompt, attention_mask=attention_mask, **gen_kwargs)
        if devices:
            _sync(devices)
        seconds.append(time.perf_counter() - t0)
        produced = int(out.shape[1]) - prompt_len
        if produced != new_tokens:
            raise RuntimeError(f"generated {produced} tokens instead of {new_tokens}")

    tok_s = [batch * new_tokens / s for s in seconds]
    peak_bytes = sum(max(0, int(torch.cuda.max_memory_allocated(d)) - baseline_allocated.get(d, 0)) for d in devices)
    return {
        "protocol": {"prompt_len": prompt_len, "new_tokens": new_tokens, "batch": batch, "runs": runs, "warmup": warmup,
                     "decoding": "greedy", "prompt": f"random token ids, seed {seed}"},
        "tokens_per_s_mean": statistics.fmean(tok_s),
        "tokens_per_s_std": statistics.stdev(tok_s) if len(tok_s) > 1 else 0.0,
        "tokens_per_s_median": statistics.median(tok_s),
        "tokens_per_s_min": min(tok_s),
        "tokens_per_s_max": max(tok_s),
        "seconds_per_run_mean": statistics.fmean(seconds),
        "weights_allocated_gb": weights_bytes / 1e9 if devices else None,
        "peak_allocated_gb": peak_bytes / 1e9 if devices else None,
        "devices": devices,
        "parameters": count_parameters(base)["structural"],
        "hardware": hardware_info(),
    }
