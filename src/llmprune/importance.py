"""Importance scores computed on the calibration sample (PyTorch).

* ``taylor_head_importance`` - thesis Equation 4 / Algorithm 1 (Michel et al., 2019):
      I(h, l) = sum over batches of | sum_{tokens, channels} f * dL/df |
  where f is the output of head h of layer l *before* the output projection W_O and L is the
  language-modelling loss. With ``batch_size = 1`` (default) the absolute value is taken per sequence.
* ``block_influence`` - thesis Equation 5 / Algorithm 3. Two definitions are computed in the same
  forward pass and both are returned:
      "frobenius": mean over sequences of ||H_l - H_{l-1}||_F / ||H_{l-1}||_F      (Equation 5 as printed)
      "cosine"   : mean over sequences and tokens of 1 - cos(H_{l-1}, H_l)          (ShortGPT, Men et al. 2024)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .io_utils import progress
from .modeling import embed_device, require_mha, unwrap


def _batches(seqs: torch.Tensor, batch_size: int):
    for i in range(0, seqs.size(0), batch_size):
        yield seqs[i:i + batch_size]


def taylor_head_importance(model, calib: torch.Tensor, batch_size: int = 1, show_progress: bool = True) -> np.ndarray:
    """First-order Taylor importance of every attention head, array ``[n_layers, n_heads]`` (float64)."""
    base = unwrap(model)
    require_mha(base)
    layers = base.model.layers
    n_layers = len(layers)
    n_heads = base.config.num_attention_heads
    head_dim = base.config.hidden_size // n_heads
    scores = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def pre_hook(module, args):
            x = args[0]
            if x.requires_grad:
                x.retain_grad()
            captured[idx] = x
        return pre_hook

    handles = [layer.self_attn.o_proj.register_forward_pre_hook(make_hook(i)) for i, layer in enumerate(layers)]
    saved_flags = [(p, p.requires_grad) for p in base.parameters()]
    was_training = base.training
    for p, _ in saved_flags:
        p.requires_grad_(False)  # gradients w.r.t. the activations are enough; saves a copy of the model in memory
    base.eval()
    emb = base.get_input_embeddings()
    try:
        with torch.enable_grad():
            for batch in progress(list(_batches(calib, batch_size)), desc="Taylor head importance", disable=not show_progress):
                ids = batch.to(embed_device(base))
                with torch.no_grad():
                    inputs_embeds = emb(ids)
                inputs_embeds = inputs_embeds.detach().requires_grad_(True)
                captured.clear()
                out = base(inputs_embeds=inputs_embeds, labels=ids, use_cache=False)
                out.loss.backward()
                for i in range(n_layers):
                    x, g = captured[i], captured[i].grad
                    if g is None:
                        raise RuntimeError(f"no gradient reached the input of o_proj in layer {i}")
                    per_head = (x.detach().float() * g.float()).reshape(x.shape[0], x.shape[1], n_heads, head_dim).sum(dim=(0, 1, 3))
                    scores[i] += per_head.abs().double().cpu()
                captured.clear()
    finally:
        for h in handles:
            h.remove()
        for p, flag in saved_flags:
            p.requires_grad_(flag)
        base.train(was_training)
    return scores.numpy()


@torch.no_grad()
def block_influence(model, calib: torch.Tensor, batch_size: int = 4, show_progress: bool = True) -> Dict[str, List[float]]:
    """Block Influence of every decoder block under both definitions (see module docstring)."""
    base = unwrap(model)
    layers = base.model.layers
    n_layers = len(layers)
    frob = torch.zeros(n_layers, dtype=torch.float64)
    cosd = torch.zeros(n_layers, dtype=torch.float64)

    def make_hook(idx: int):
        def hook(module, args, kwargs, output):
            xi = args[0] if len(args) > 0 else kwargs["hidden_states"]
            xo = output[0] if isinstance(output, (tuple, list)) else output
            xi = xi.detach().to(xo.device).float()  # accelerate leaves the input on the previous layer's device
            xo = xo.detach().float()
            b = xo.shape[0]
            ratio = (xo - xi).reshape(b, -1).norm(dim=1) / xi.reshape(b, -1).norm(dim=1).clamp_min(1e-12)
            frob[idx] += ratio.sum().double().cpu()
            cos = F.cosine_similarity(xi, xo, dim=-1, eps=1e-8)  # [b, tokens]
            cosd[idx] += (1.0 - cos).mean(dim=1).sum().double().cpu()
        return hook

    handles = [layer.register_forward_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    was_training = base.training
    base.eval()
    try:
        for batch in progress(list(_batches(calib, batch_size)), desc="Block influence", disable=not show_progress):
            base.model(input_ids=batch.to(embed_device(base)), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        base.train(was_training)
    n = calib.size(0)
    return {"frobenius": (frob / n).tolist(), "cosine": (cosd / n).tolist()}
