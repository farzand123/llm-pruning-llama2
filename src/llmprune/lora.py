"""LoRA recovery (PyTorch + PEFT) - thesis Section III.D.4 and Table 11.

A hand-written training loop is used instead of ``transformers.Trainer`` because it makes the three things
the thesis states explicitly unambiguous: the effective batch is exactly micro-batch x accumulation
(Trainer would multiply it by the number of visible GPUs under DataParallel), the frozen base stays in
bfloat16 while the adapters are trained in float32, and the base weights are verified to be unchanged
(so that the zeros created by magnitude pruning provably stay zero, Algorithm 2 step 4).
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .data import build_alpaca, make_collate
from .io_utils import ensure_dir, utc_now, write_json
from .modeling import embed_device, fingerprint, set_seed

log = logging.getLogger(__name__)


def attach_lora(model, rank: int, alpha: int, dropout: float, target_modules: Sequence[str]):
    """Wrap ``model`` with LoRA adapters. The adapters are trained in float32 even if the base is bfloat16
    (PEFT 0.10 would otherwise cast them to the base dtype)."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=int(rank), lora_alpha=int(alpha), lora_dropout=float(dropout),
                        target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, config)
    for p in peft_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    peft_model.config.use_cache = False
    return peft_model


def load_adapter(model, adapter_dir):
    """Attach a trained adapter for inference (kept unmerged) and run it in float32 as in training.

    PEFT 0.10 creates the adapter modules in the base dtype, so values loaded from the float32 file may be
    rounded to bfloat16 before the upcast (at most ~0.4 % relative per weight, negligible next to bf16 base weights).
    """
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    for name, p in peft_model.named_parameters():
        if "lora_" in name:
            p.data = p.data.float()
    peft_model.eval()
    return peft_model


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier: linear warm-up, then cosine decay to zero."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(model, loader: DataLoader, *, lr: float, weight_decay: float, epochs: int, grad_accum: int,
          warmup_steps: int, max_grad_norm: Optional[float], max_steps: Optional[int] = None,
          log_every: int = 10, log_path: Optional[Path] = None) -> Dict[str, Any]:
    """Train the adapters of a PEFT model. One optimiser step = ``grad_accum`` micro-batches."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("the model has no trainable parameters")
    if len(loader) == 0:
        raise RuntimeError("the data loader is empty (fewer examples than one micro-batch?)")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, int(max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total_steps, warmup_steps))
    dev = embed_device(model)

    step = micro = tokens = 0
    window_loss, window_n = 0.0, 0
    history: List[List[float]] = []
    t0 = time.time()
    log_fh = open(log_path, "w") if log_path else None

    def optimizer_step() -> None:
        nonlocal step, window_loss, window_n
        if window_n != grad_accum:  # short window at the end of an epoch: keep the gradient an average
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(grad_accum / window_n)
        gnorm = None
        if max_grad_norm:
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, float(max_grad_norm)))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        mean_loss = window_loss / window_n
        history.append([step, mean_loss])
        window_loss, window_n = 0.0, 0
        if log_fh is not None:
            log_fh.write(json.dumps({"step": step, "loss": mean_loss, "lr": sched.get_last_lr()[0], "grad_norm": gnorm,
                                     "elapsed_s": round(time.time() - t0, 1)}) + "\n")
            log_fh.flush()
        if step % log_every == 0 or step == total_steps:
            log.info("step %d/%d  loss %.4f  lr %.2e  %.0fs", step, total_steps, mean_loss, sched.get_last_lr()[0], time.time() - t0)

    model.train()
    try:
        for _epoch in range(epochs):
            for batch in loader:
                batch = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    out = model(**batch)
                loss = out.loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"training loss is {float(loss)} at optimiser step {step + 1}")
                (loss / grad_accum).backward()
                window_loss += float(loss.detach())
                window_n += 1
                micro += 1
                tokens += int(batch["attention_mask"].sum())
                if window_n == grad_accum:
                    optimizer_step()
                    if max_steps and step >= max_steps:
                        break
            else:
                if window_n > 0:
                    optimizer_step()
                if max_steps and step >= max_steps:
                    break
                continue
            break  # inner loop was left through ``break`` (max_steps reached)
    finally:
        if log_fh is not None:
            log_fh.close()
        model.eval()

    losses = [h[1] for h in history]
    keep_every = max(1, len(history) // 200)
    return {
        "steps": step, "planned_steps": total_steps, "micro_batches": micro, "tokens_seen": tokens,
        "first_loss": sum(losses[:10]) / max(1, len(losses[:10])),
        "final_loss": sum(losses[-50:]) / max(1, len(losses[-50:])),
        "loss_curve": history[::keep_every], "elapsed_s": round(time.time() - t0, 1),
    }


def train_lora(cfg: Dict[str, Any], cond, model, tokenizer, out_dir, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Train and save the LoRA adapter of one condition on the (already pruned) ``model``."""
    lc = cfg["lora"]
    seed = int(cfg["seed"])
    out_dir = ensure_dir(out_dir)
    set_seed(seed)
    rank = int(cond.lora_rank)
    alpha = int(lc["alpha_multiplier"] * rank)

    before = fingerprint(model)
    peft_model = attach_lora(model, rank, alpha, lc["dropout"], lc["target_modules"])
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    log.info("LoRA r=%d alpha=%d: %.2f M trainable of %.3f B parameters", rank, alpha, trainable / 1e6, total / 1e9)

    dataset = build_alpaca(tokenizer, int(lc["max_len"]), lc["dataset"], lc.get("max_examples"),
                           bool(lc.get("train_on_inputs", False)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(lc["micro_batch_size"]), shuffle=True, drop_last=True,
                        collate_fn=make_collate(tokenizer.pad_token_id), generator=gen)
    stats = train(peft_model, loader, lr=float(lc["learning_rate"]), weight_decay=float(lc["weight_decay"]),
                  epochs=int(lc["epochs"]), grad_accum=int(lc["grad_accum"]), warmup_steps=int(lc.get("warmup_steps", 100)),
                  max_grad_norm=lc.get("max_grad_norm", 1.0), max_steps=lc.get("max_steps"),
                  log_path=Path(out_dir) / "train_log.jsonl")

    after = fingerprint(peft_model)
    unchanged = (before["nonzero_prunable"] == after["nonzero_prunable"]
                 and math.isclose(before["abs_sum_prunable"], after["abs_sum_prunable"], rel_tol=1e-9))
    if not unchanged:
        log.error("base weights changed during LoRA training: %s -> %s", before, after)
    peft_model.save_pretrained(str(out_dir))
    summary = {
        "rank": rank, "alpha": alpha, "dropout": float(lc["dropout"]), "target_modules": list(lc["target_modules"]),
        "trainable_params": int(trainable), "total_params": int(total),
        "effective_batch": int(lc["micro_batch_size"]) * int(lc["grad_accum"]),
        "n_examples": len(dataset), "lr": float(lc["learning_rate"]), "weight_decay": float(lc["weight_decay"]),
        "epochs": int(lc["epochs"]), "seed": seed, "max_len": int(lc["max_len"]), **stats,
        "base_weights_unchanged": bool(unchanged), "base_fingerprint_before": before, "base_fingerprint_after": after,
        "finished": utc_now(), "provenance": provenance,
    }
    write_json(Path(out_dir) / "train_summary.json", summary)  # written last: its existence marks the stage as done
    return summary
