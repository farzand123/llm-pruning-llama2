"""Analytical parameter, FLOPs, KV-cache and memory accounting (pure Python).

These formulas reproduce thesis Equations 9-13 and 15 and Tables 14, 21 and 22. They
need only the architecture constants, so they run anywhere and serve as an independent
check of what the PyTorch code measures.

Conventions
-----------
* "structural" parameters = every stored element (what ``sum(p.numel())`` gives).
* Attention is multi-head (LLaMA-2-7B/13B). Grouped-query attention is not supported.
* ``heads_per_layer`` is a list with the number of *retained* heads of each layer.
* FLOPs per generated token follow Equation 13:  F = 2 * N_lin + 4 * n_ctx * sum_l(d_attn_l)
  where N_lin counts the parameters of all linear layers including the output head and
  d_attn_l = h_l * head_dim is the width of the retained heads of layer l.
* Memory is in decimal gigabytes (1 GB = 1e9 bytes), as in thesis Table 22.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Arch:
    hidden: int
    ffn: int
    layers: int
    heads: int
    vocab: int
    kv_heads: Optional[int] = None
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def check_mha(self) -> None:
        if self.kv_heads not in (None, self.heads):
            raise NotImplementedError("grouped-query attention is not supported")


LLAMA2_7B = Arch(hidden=4096, ffn=11008, layers=32, heads=32, vocab=32000)
LLAMA2_13B = Arch(hidden=5120, ffn=13824, layers=40, heads=40, vocab=32000)


def arch_from_config(cfg) -> Arch:
    """Build an Arch from a Hugging Face ``LlamaConfig`` (or any object with these attributes)."""
    return Arch(
        hidden=int(cfg.hidden_size),
        ffn=int(cfg.intermediate_size),
        layers=int(cfg.num_hidden_layers),
        heads=int(cfg.num_attention_heads),
        vocab=int(cfg.vocab_size),
        kv_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        tied_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
    )


def uniform_heads(arch: Arch, n_layers: Optional[int] = None) -> List[int]:
    return [arch.heads] * (arch.layers if n_layers is None else n_layers)


def heads_per_layer_from_total(arch: Arch, n_kept_total: int) -> List[int]:
    """Spread ``n_kept_total`` retained heads as evenly as possible (analytical use only:
    parameters, FLOPs and KV size depend only on the total number of retained heads)."""
    base, extra = divmod(n_kept_total, arch.layers)
    return [base + (1 if l < extra else 0) for l in range(arch.layers)]


def _resolve(arch: Arch, heads_per_layer, n_layers):
    arch.check_mha()
    if heads_per_layer is None:
        return uniform_heads(arch, n_layers)
    if n_layers is not None and n_layers != len(heads_per_layer):
        raise ValueError("n_layers disagrees with len(heads_per_layer)")
    return list(heads_per_layer)


def attention_params(arch: Arch, h: int) -> int:
    """Q, K, V and O projections of one layer that keeps ``h`` heads (Equation 15 per head)."""
    return 4 * arch.hidden * h * arch.head_dim


def ffn_params(arch: Arch) -> int:
    return 3 * arch.hidden * arch.ffn


def embedding_params(arch: Arch) -> int:
    return arch.vocab * arch.hidden * (1 if arch.tied_embeddings else 2)


def total_params(arch: Arch, heads_per_layer: Optional[Sequence[int]] = None, n_layers: Optional[int] = None) -> int:
    """Structural parameters: embeddings + output head + final norm + all blocks."""
    heads = _resolve(arch, heads_per_layer, n_layers)
    blocks = sum(attention_params(arch, h) + ffn_params(arch) + 2 * arch.hidden for h in heads)
    return embedding_params(arch) + arch.hidden + blocks


def linear_params(arch: Arch, heads_per_layer: Optional[Sequence[int]] = None, n_layers: Optional[int] = None) -> int:
    """N_lin of Equation 13: all linear layers, including the output head."""
    heads = _resolve(arch, heads_per_layer, n_layers)
    return sum(attention_params(arch, h) + ffn_params(arch) for h in heads) + arch.vocab * arch.hidden


def prunable_params(arch: Arch, n_layers: Optional[int] = None) -> int:
    """Weights magnitude pruning may zero: Q, K, V, O and the three FFN matrices of every block."""
    n = arch.layers if n_layers is None else n_layers
    return n * (attention_params(arch, arch.heads) + ffn_params(arch))


def flops_per_token(
    arch: Arch,
    heads_per_layer: Optional[Sequence[int]] = None,
    n_layers: Optional[int] = None,
    ctx: int = 2048,
) -> int:
    """Equation 13. Dense execution is assumed (zeros still cost FLOPs)."""
    heads = _resolve(arch, heads_per_layer, n_layers)
    d_attn_sum = sum(h * arch.head_dim for h in heads)
    return 2 * linear_params(arch, heads_per_layer, n_layers) + 4 * ctx * d_attn_sum


def kv_cache_bytes_per_token(
    arch: Arch,
    heads_per_layer: Optional[Sequence[int]] = None,
    n_layers: Optional[int] = None,
    bytes_per_element: int = 2,
) -> float:
    heads = _resolve(arch, heads_per_layer, n_layers)
    return 2.0 * sum(h * arch.head_dim for h in heads) * bytes_per_element


def weight_gb(n_params: float, bytes_per_param: int = 2) -> float:
    return n_params * bytes_per_param / 1e9


def lora_trainable_params(
    arch: Arch,
    rank: int,
    heads_per_layer: Optional[Sequence[int]] = None,
    n_layers: Optional[int] = None,
) -> int:
    """LoRA on Q, K, V, O: each adapter has r * (in + out) parameters.

    Per layer: 3 * r * (d + h*hd) for Q, K, V and r * (h*hd + d) for O = 4 * r * (d + h*hd).
    """
    heads = _resolve(arch, heads_per_layer, n_layers)
    return sum(4 * rank * (arch.hidden + h * arch.head_dim) for h in heads)


def reduction_pct(orig: float, pruned: float) -> float:
    """Equations 10 and 11: 100 * (orig - pruned) / orig."""
    return 100.0 * (orig - pruned) / orig


def compression_ratio(orig: float, pruned: float) -> float:
    """Equation 9: orig / pruned."""
    return orig / pruned


def profile(
    arch: Arch,
    heads_per_layer: Optional[Sequence[int]] = None,
    n_layers: Optional[int] = None,
    ctx: int = 2048,
) -> dict:
    """One-stop analytical profile of a (possibly pruned) architecture."""
    base_params = total_params(arch)
    base_flops = flops_per_token(arch, ctx=ctx)
    p = total_params(arch, heads_per_layer, n_layers)
    f = flops_per_token(arch, heads_per_layer, n_layers, ctx)
    return {
        "structural_params": p,
        "param_reduction_pct": reduction_pct(base_params, p),
        "cr_struct": compression_ratio(base_params, p),
        "flops_per_token": f,
        "flops_reduction_pct": reduction_pct(base_flops, f),
        "weight_gb_bf16": weight_gb(p),
        "kv_bytes_per_token_bf16": kv_cache_bytes_per_token(arch, heads_per_layer, n_layers),
    }
