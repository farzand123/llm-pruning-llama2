"""Datasets (PyTorch + Hugging Face ``datasets``).

* C4 calibration sample  - thesis III.B.4: 128 sequences x 2,048 tokens, seed 42. The sampling rule is the
  one used by SparseGPT and Wanda (random document from one C4 shard, random window of ``seqlen`` tokens
  from a document that is longer than ``seqlen``).
* WikiText-2 test split  - the standard "\\n\\n".join(...) evaluation text.
* Alpaca-GPT4            - LoRA recovery data (Table 11), Stanford-Alpaca prompt template.
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger(__name__)

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


# --------------------------------------------------------------------------- calibration (C4)
def _load_c4(shard: str, local_path: Optional[str]):
    from datasets import load_dataset

    if local_path:
        return load_dataset("json", data_files={"train": str(local_path)}, split="train")
    errors = []
    for kwargs in ({}, {"name": "allenai--c4"}):  # the second form is what older SparseGPT/Wanda code uses
        try:
            return load_dataset("allenai/c4", data_files={"train": shard}, split="train", **kwargs)
        except Exception as exc:  # noqa: BLE001 - report every attempt
            errors.append(f"{kwargs or 'default config'}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "could not load the C4 shard from the Hugging Face Hub. Download "
        f"{shard} from https://huggingface.co/datasets/allenai/c4 and set calibration.c4_local_path. "
        "Attempts: " + " | ".join(errors)
    )


def calibration_hash(calib: torch.Tensor) -> str:
    return hashlib.sha256(calib.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()


def get_c4_calibration(tokenizer, n_samples: int, seqlen: int, seed: int,
                       shard: str = "en/c4-train.00000-of-01024.json.gz",
                       local_path: Optional[str] = None,
                       cache_dir: Optional[str] = None) -> torch.Tensor:
    """LongTensor ``[n_samples, seqlen]``. Deterministic for a given (seed, shard, tokenizer)."""
    key = hashlib.sha1(
        f"{shard}|{local_path}|{n_samples}|{seqlen}|{seed}|{tokenizer.name_or_path}|{len(tokenizer)}".encode()
    ).hexdigest()[:16]
    cache = Path(cache_dir) / f"calibration_{key}.pt" if cache_dir else None
    if cache is not None and cache.is_file():
        log.info("calibration sample loaded from %s", cache)
        return torch.load(cache)

    ds = _load_c4(shard, local_path)
    rng = random.Random(seed)
    n_docs = len(ds)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        for _attempt in range(10_000):
            ids = tokenizer(ds[rng.randint(0, n_docs - 1)]["text"], return_tensors="pt").input_ids
            if ids.shape[1] > seqlen:
                break
        else:  # pragma: no cover
            raise RuntimeError(f"no C4 document longer than {seqlen} tokens found")
        start = rng.randint(0, ids.shape[1] - seqlen - 1)
        samples.append(ids[0, start:start + seqlen].clone())
    calib = torch.stack(samples)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(calib, cache)
    return calib


# --------------------------------------------------------------------------- WikiText-2
def get_wikitext2_test_ids(tokenizer) -> torch.Tensor:
    """Token ids ``[1, N]`` of the WikiText-2 (raw) test split joined with blank lines."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids


# --------------------------------------------------------------------------- Alpaca-GPT4
def format_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    if input_text and str(input_text).strip():
        return ALPACA_WITH_INPUT.format(instruction=instruction, input=input_text)
    return ALPACA_NO_INPUT.format(instruction=instruction)


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: List[List[int]], labels: List[List[int]]):
        assert len(input_ids) == len(labels)
        self.input_ids, self.labels = input_ids, labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def encode_example(tokenizer, prompt: str, output: str, max_len: int, train_on_inputs: bool = False):
    """(input_ids, labels) of one instruction example, or None if no response token survives truncation."""
    ids = list(tokenizer(prompt + output, truncation=True, max_length=max_len)["input_ids"])
    p_ids = tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"]
    if len(ids) < max_len and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    n_prompt = min(len(p_ids), len(ids))
    labels = list(ids) if train_on_inputs else [-100] * n_prompt + ids[n_prompt:]
    if all(x == -100 for x in labels):
        return None
    return ids, labels


def build_alpaca(tokenizer, max_len: int = 512, dataset_name: str = "vicgalle/alpaca-gpt4",
                 max_examples: Optional[int] = None, train_on_inputs: bool = False,
                 chunk: int = 2000) -> SupervisedDataset:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    n = len(ds) if max_examples is None else min(int(max_examples), len(ds))
    ids_all: List[List[int]] = []
    labels_all: List[List[int]] = []
    skipped = 0
    for start in range(0, n, chunk):
        rows = ds[start:min(start + chunk, n)]
        for ins, inp, out in zip(rows["instruction"], rows["input"], rows["output"]):
            enc = encode_example(tokenizer, format_prompt(ins, inp), out, max_len, train_on_inputs)
            if enc is None:
                skipped += 1
                continue
            ids_all.append(enc[0])
            labels_all.append(enc[1])
    log.info("%s: %d examples tokenised, %d skipped (no response token within %d tokens)",
             dataset_name, len(ids_all), skipped, max_len)
    return SupervisedDataset(ids_all, labels_all)


def make_collate(pad_id: int):
    """Right-padding collate function producing input_ids, attention_mask and labels."""
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), width), -100, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = b["input_ids"]
            lab[i, :n] = b["labels"]
            att[i, :n] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate
