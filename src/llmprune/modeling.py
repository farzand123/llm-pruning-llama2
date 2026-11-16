"""Model loading and the structural edits behind the three pruning methods (PyTorch).

None of the three methods changes a surviving weight:

* heads     - rows of W_Q/W_K/W_V and columns of W_O are *sliced out* (Algorithm 1),
* layers    - decoder blocks are *deleted* (Algorithm 3),
* magnitude - weights whose magnitude bit pattern is below a stored threshold are zeroed (Algorithm 2).

A small JSON "recipe" plus the base model therefore reproduces a pruned model exactly (see
``pruning.py``); no multi-gigabyte pruned checkpoints are needed.

Written against transformers 4.38.2 (thesis Table 10). Attribute names that changed in later
releases are set only if they exist, and head pruning is verified against a masked reference.
"""
from __future__ import annotations

import gc
import logging
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import selection as sel

log = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------- utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unwrap(model):
    """The underlying ``LlamaForCausalLM`` of a PEFT-wrapped model (identity otherwise)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available(), "n_gpus": torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                        for i in range(torch.cuda.device_count())]
    return info


# --------------------------------------------------------------------------- loading
def resolve_device_map(device_map):
    """'auto' -> accelerate spreads consecutive layers over all visible GPUs (the thesis setting);
    'single' -> everything on GPU 0; 'cpu'/None -> CPU; a dict is passed through.
    Without CUDA every setting falls back to the CPU."""
    if device_map in (None, "cpu"):
        return None
    if not torch.cuda.is_available():
        log.warning("device_map=%r requested but CUDA is not available; using the CPU", device_map)
        return None
    if device_map == "single":
        return {"": 0}
    return device_map


def load_model(name_or_path: str, dtype: str = "bfloat16", device_map="auto",
               revision: Optional[str] = None, attn_implementation: Optional[str] = None):
    """Load a causal LM and its tokenizer. ``name_or_path`` may be a Hub id or a local directory."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    rev = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, **rev)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    config = AutoConfig.from_pretrained(name_or_path, **rev)
    if getattr(config, "pretraining_tp", 1) not in (None, 1):
        # The tp > 1 branch of LlamaAttention/LlamaForCausalLM calls F.linear on weight slices and
        # therefore bypasses the nn.Linear modules: LoRA would silently have no effect and head
        # pruning would see sliced weights. LLaMA-2-13B ships with pretraining_tp = 2.
        log.info("config.pretraining_tp %s -> 1", config.pretraining_tp)
        config.pretraining_tp = 1
    kwargs: Dict[str, Any] = dict(config=config, torch_dtype=DTYPES[dtype], low_cpu_mem_usage=True, **rev)
    dm = resolve_device_map(device_map)
    if dm is not None:
        kwargs["device_map"] = dm
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    model.eval()
    return model, tok


def embed_device(model) -> torch.device:
    """Device that must hold the input ids (works for plain and PEFT-wrapped models)."""
    return model.get_input_embeddings().weight.device


def require_mha(model_or_config) -> None:
    cfg = getattr(unwrap(model_or_config), "config", model_or_config)
    kv = getattr(cfg, "num_key_value_heads", None)
    if kv not in (None, cfg.num_attention_heads):
        raise NotImplementedError(
            f"grouped-query attention (num_key_value_heads={kv}, num_attention_heads={cfg.num_attention_heads}) "
            "is not supported; the thesis uses LLaMA-2-7B/13B, which have multi-head attention"
        )


# --------------------------------------------------------------------------- inspection
@torch.no_grad()
def count_parameters(model) -> Dict[str, int]:
    """Structural (stored), non-zero and byte counts of all parameters (thesis Table 21)."""
    structural = nonzero = nbytes = 0
    for p in model.parameters():
        structural += p.numel()
        nonzero += int(torch.count_nonzero(p).item())
        nbytes += p.numel() * p.element_size()
    return {"structural": int(structural), "nonzero": int(nonzero), "weight_bytes": int(nbytes)}


def describe_shapes(model) -> Dict[str, Any]:
    """Tensor shapes of every block plus head counts (thesis Table 14)."""
    base = unwrap(model)
    cfg = base.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers, attn_params, ffn_params, kv_bytes = [], 0, 0, 0
    for layer in base.model.layers:
        a, m = layer.self_attn, layer.mlp
        entry = {n: list(getattr(a, n).weight.shape) for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
        entry.update({n: list(getattr(m, n).weight.shape) for n in ("gate_proj", "up_proj", "down_proj")})
        layers.append(entry)
        attn_params += sum(getattr(a, n).weight.numel() for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
        ffn_params += sum(getattr(m, n).weight.numel() for n in ("gate_proj", "up_proj", "down_proj"))
        kv_bytes += 2 * a.k_proj.out_features * a.k_proj.weight.element_size()
    return {
        "n_layers": len(layers),
        "n_heads_total": sum(l["q_proj"][0] for l in layers) // head_dim,
        "head_dim": head_dim,
        "layers": layers,
        "attention_params": int(attn_params),
        "ffn_params": int(ffn_params),
        "kv_bytes_per_token": int(kv_bytes),
    }


# --------------------------------------------------------------------------- attention heads
@torch.no_grad()
def _slice_linear(old: nn.Linear, rows: Optional[torch.Tensor] = None, cols: Optional[torch.Tensor] = None) -> nn.Linear:
    """New Linear holding ``old.weight[rows][:, cols]`` on the same device and dtype."""
    w = old.weight
    out_f = old.out_features if rows is None else int(rows.numel())
    in_f = old.in_features if cols is None else int(cols.numel())
    new = torch.nn.utils.skip_init(nn.Linear, in_f, out_f, bias=old.bias is not None, device=w.device, dtype=w.dtype)
    src = w
    if rows is not None:
        src = src[rows.to(w.device)]
    if cols is not None:
        src = src[:, cols.to(w.device)]
    new.weight.copy_(src)
    if old.bias is not None:
        new.bias.copy_(old.bias[rows.to(w.device)] if rows is not None else old.bias)
        new.bias.requires_grad_(old.bias.requires_grad)
    new.weight.requires_grad_(w.requires_grad)
    return new


def _set_attention_attrs(att: nn.Module, n_heads: int, head_dim: int) -> None:
    """Keep the attention module's bookkeeping consistent with its new width.

    transformers 4.38 reshapes with ``num_heads`` and ``hidden_size``; later versions infer the
    width from the tensor. Only attributes that exist are touched.
    """
    for attr, value in (("num_heads", n_heads), ("num_key_value_heads", n_heads),
                        ("hidden_size", n_heads * head_dim), ("num_key_value_groups", 1)):
        if hasattr(att, attr):
            setattr(att, attr, value)


@torch.no_grad()
def apply_head_pruning(model, keep_per_layer: Sequence[Sequence[int]]) -> List[int]:
    """Algorithm 1: physically remove every head not listed in ``keep_per_layer``."""
    base = unwrap(model)
    require_mha(base)
    cfg = base.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers = base.model.layers
    if len(keep_per_layer) != len(layers):
        raise ValueError(f"recipe has {len(keep_per_layer)} layers, model has {len(layers)}")
    for layer, keep in zip(layers, keep_per_layer):
        if len(keep) == 0:
            raise ValueError("a layer cannot lose all of its heads")
        idx = torch.as_tensor(sel.head_channel_indices(keep, head_dim), dtype=torch.long)
        att = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(att, name, _slice_linear(getattr(att, name), rows=idx))
        att.o_proj = _slice_linear(att.o_proj, cols=idx)
        _set_attention_attrs(att, len(keep), head_dim)
    return [len(k) for k in keep_per_layer]


@torch.no_grad()
def logits_with_heads_masked(model, remove_mask: np.ndarray, ids: torch.Tensor) -> torch.Tensor:
    """Reference for the verification step: logits of the *unpruned* model with the to-be-removed
    heads switched off by zeroing their columns of W_O (restored afterwards)."""
    base = unwrap(model)
    head_dim = base.config.hidden_size // base.config.num_attention_heads
    saved = []
    for layer, row in zip(base.model.layers, np.asarray(remove_mask, dtype=bool)):
        heads = np.flatnonzero(row)
        if heads.size == 0:
            continue
        W = layer.self_attn.o_proj.weight
        cols = torch.as_tensor(sel.head_channel_indices(heads.tolist(), head_dim), dtype=torch.long, device=W.device)
        saved.append((W, cols, W[:, cols].clone()))
        W[:, cols] = 0
    try:
        logits = model(ids.to(embed_device(model)), use_cache=False).logits.float().cpu()
    finally:
        for W, cols, original in saved:
            W[:, cols] = original
    return logits


@torch.no_grad()
def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    ref, cand = reference.float().cpu(), candidate.float().cpu()
    return {
        "rel_l2": float((ref - cand).norm() / ref.norm().clamp_min(1e-12)),
        "max_abs": float((ref - cand).abs().max()),
        "argmax_agreement": float((ref.argmax(-1) == cand.argmax(-1)).float().mean()),
    }


# --------------------------------------------------------------------------- layers
def _renumber_device_map(device_map: Dict[str, Any], dropped: set) -> Dict[str, Any]:
    pattern = re.compile(r"model\.layers\.(\d+)")
    old_ids = sorted({int(m.group(1)) for m in map(pattern.fullmatch, device_map) if m})
    new_id = {old: new for new, old in enumerate(i for i in old_ids if i not in dropped)}
    out: Dict[str, Any] = {}
    for key, dev in device_map.items():
        m = pattern.fullmatch(key)
        if m is None:
            out[key] = dev
        elif int(m.group(1)) in new_id:
            out[f"model.layers.{new_id[int(m.group(1))]}"] = dev
    return out


@torch.no_grad()
def remove_layers(model, indices: Sequence[int]) -> List[int]:
    """Algorithm 3: delete decoder blocks and renumber the rest so that the KV cache stays aligned."""
    base = unwrap(model)
    layers = base.model.layers
    drop = sorted({int(i) for i in indices})
    if any(i < 0 or i >= len(layers) for i in drop):
        raise ValueError(f"layer index out of range: {drop}")
    if len(drop) >= len(layers):
        raise ValueError("cannot remove every layer")
    for i in reversed(drop):
        del layers[i]
    for new_idx, layer in enumerate(layers):
        for obj in (layer, layer.self_attn):
            if hasattr(obj, "layer_idx"):
                obj.layer_idx = new_idx
    base.config.num_hidden_layers = len(layers)
    layer_types = getattr(base.config, "layer_types", None)
    if layer_types:
        base.config.layer_types = [t for i, t in enumerate(layer_types) if i not in set(drop)]
    if getattr(base, "hf_device_map", None):
        base.hf_device_map = _renumber_device_map(base.hf_device_map, set(drop))
    return drop


# --------------------------------------------------------------------------- magnitude
_PATTERN_MASK = 0x7FFF  # magnitude bits of a 16-bit float (sign bit removed)


def prunable_linears(model) -> List[tuple]:
    """Q, K, V, O and the three FFN matrices of every block (embeddings, output head and norms excluded)."""
    mods = []
    for i, layer in enumerate(unwrap(model).model.layers):
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mods.append((f"layers.{i}.self_attn.{n}", getattr(layer.self_attn, n)))
        for n in ("gate_proj", "up_proj", "down_proj"):
            mods.append((f"layers.{i}.mlp.{n}", getattr(layer.mlp, n)))
    return mods


def _base_weight(module) -> torch.Tensor:
    """Weight of an nn.Linear, looking through a PEFT LoRA wrapper if present."""
    return getattr(module, "base_layer", module).weight


def _magnitude_pattern(w: torch.Tensor) -> torch.Tensor:
    if w.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError("magnitude pruning needs bfloat16 or float16 weights (set runtime.dtype)")
    return w.detach().contiguous().view(torch.int16) & _PATTERN_MASK


@torch.no_grad()
def magnitude_histogram(model) -> np.ndarray:
    """Number of prunable weights per magnitude bit pattern (32,768 bins), accumulated exactly."""
    hist = np.zeros(_PATTERN_MASK + 1, dtype=np.int64)
    for _, m in prunable_linears(model):
        pat = _magnitude_pattern(_base_weight(m)).reshape(-1).to(torch.int64)
        hist += torch.bincount(pat, minlength=_PATTERN_MASK + 1).cpu().numpy()
    return hist


@torch.no_grad()
def apply_pattern_threshold(model, pattern_threshold: int) -> int:
    """Zero every prunable weight whose magnitude pattern is below the threshold; returns the count."""
    zeroed = 0
    for _, m in prunable_linears(model):
        w = _base_weight(m)
        below = _magnitude_pattern(w) < int(pattern_threshold)
        zeroed += int(below.sum().item())
        w.masked_fill_(below, 0.0)
    return zeroed


@torch.no_grad()
def magnitude_prune(model, ratio: float) -> Dict[str, Any]:
    """Algorithm 2: global magnitude pruning with an exact threshold (no sampling, no sorting)."""
    hist = magnitude_histogram(model)
    total = int(hist.sum())
    target = sel.n_to_remove(ratio, total)
    p_star, expected, _ = sel.threshold_from_histogram(hist, target)
    zeroed = apply_pattern_threshold(model, p_star)
    if zeroed != expected:  # pragma: no cover - cannot happen unless weights changed in between
        raise RuntimeError(f"zeroed {zeroed} weights but the histogram predicted {expected}")
    return {
        "pattern_threshold": int(p_star),
        "prunable_weights": total,
        "target_zeroed": int(target),
        "zeroed": int(zeroed),
        "realised_ratio": zeroed / total,
        "threshold_method": "exact histogram of 15-bit magnitude patterns",
    }


# --------------------------------------------------------------------------- fingerprints
@torch.no_grad()
def fingerprint(model) -> Dict[str, Any]:
    """Cheap evidence that base weights were not modified. Used around LoRA training, so that the
    zeros created by magnitude pruning are verified to still be zero (Algorithm 2, step 4)."""
    nnz = 0
    abs_sum = 0.0
    for _, m in prunable_linears(model):
        w = _base_weight(m).detach()
        nnz += int(torch.count_nonzero(w).item())
        abs_sum += float(w.float().abs().sum().item())
    return {"nonzero_prunable": nnz, "abs_sum_prunable": abs_sum}
