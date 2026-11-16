# Pruning Large Language Models — reproduction code

Code for the Master's thesis *Pruning Large Language Models* (Muhammad Farzand, Istanbul Aydın University,
Institute of Graduate Studies, 2026). It compares three ways of shrinking **LLaMA-2-7B** (and, for the
cross-scale efficiency check, LLaMA-2-13B), each with and without LoRA recovery:

| Method  |
|---|---|---|
| Global magnitude pruning | individual weights of Q, K, V, O and the FFN matrices (one global threshold) | Algorithm 2 |
| Taylor attention-head pruning | whole heads, ranked by first-order Taylor importance (rows of W_Q/W_K/W_V, columns of W_O are physically removed) | Equation 4, Algorithm 1 |
| Block-Influence layer pruning | whole decoder blocks with the smallest Block Influence | Equation 5, Algorithm 3 |

The repository runs the experiment matrix of thesis **Table 17**, evaluates every condition (WikiText-2
perplexity, 5-shot MMLU, GLUE, a generation benchmark), and regenerates **Tables 14–24** and the figures
from the result files.



## Quick start

```bash
# 1. environment (thesis Table 10: Python 3.10, PyTorch 2.2.0, transformers 4.38.2, PEFT 0.10.0, ...)
python3.10 -m venv .venv && source .venv/bin/activate
make install-dev            # installs requirements-dev.txt and the package (editable)
huggingface-cli login       # LLaMA-2 is gated: accept Meta's licence on the Hub first

# 2. does this installation work?   (no downloads, no GPU needed, ~1 min)
make selftest               # 11 checks on a tiny random LLaMA
make test                   # unit tests
make checks                 # reproduces the thesis' analytical numbers (parameters, FLOPs, KV cache, LoRA sizes)
make check                  # versions, GPUs, Hugging Face token, disk space

# 3. smoke-test the whole pipeline on a 160M model (minutes)
make quick                  # prune -> LoRA -> evaluate -> benchmark -> report, written to *_quick folders

# 4. the thesis matrix (days on 4 x A100; resumable)
make plan-full              # prints Table 17 as a list of conditions and what each needs
make full                   # or shard it over GPUs, see "Running the full matrix"
```

A step-by-step version with troubleshooting is in [docs/REPRODUCING.md](docs/REPRODUCING.md).
`llmprune <command> --config configs/full.yaml` is the underlying interface:
`plan`, `run` (aliases `prune`, `lora`, `evaluate`, `benchmark`), `status`, `report`, `aggregate`, `check`,
`selftest`. `llmprune <command> -h` lists the options (`--only`, `--shard`, `--stages`, `--force`, `--dry-run`).

## Running the full matrix

Every condition has an id such as `7b-heads-30-lora` (Table 17 number in `llmprune plan`). Select by id,
glob (`--only '7b-heads-*'`) or number (`--only 15 21`).

```bash
llmprune plan --config configs/full.yaml                      # 45 conditions: 21 recipes, 22 LoRA runs, 12 benchmarks

# four GPUs, one condition stream per GPU (device_map "auto" then sees a single GPU)
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k llmprune run --config configs/full.yaml --stages recipe,lora,eval --shard $k/4 \
      > logs_shard$k.txt 2>&1 &
done; wait

# efficiency benchmarks need idle GPUs (timing) and, for the "auto" setting, all four of them
llmprune run --config configs/full.yaml --stages efficiency

llmprune status --config configs/full.yaml --missing          # what is missing or stale
llmprune report --config configs/full.yaml                    # tables, figures, comparison with the thesis
```

Runs are **resumable**: every stage writes its file atomically when it is complete, and a rerun does only
what is missing. If an upstream file changes (a recipe is recomputed, an adapter retrained) everything
built on it is flagged *stale* and redone. Two shards that need the same recipe never compute it twice.
Do not combine `--force` with `--shard`.

## What it produces

| Folder | Content | Commit to git? |
|---|---|---|
| `recipes/<model>-<method>-<pct>.json` | the pruning decision: retained heads / removed layers / magnitude threshold, scores, parameter counts, tensor shapes before and after, calibration hash, verification, environment | **yes** (kilobytes) |
| `results/<condition>/eval_ppl.json`, `eval_mmlu.json`, `eval_glue.json`, `efficiency_<setting>.json` | raw metrics plus the evaluation protocol and provenance (git commit, library versions, hardware, timestamp) | **yes** |
| `adapters/<condition>/` | LoRA adapter, `train_summary.json`, `train_log.jsonl` (loss per optimiser step) | no (67 MB at r = 16 for 7B; publish via the Hub or a release) |
| `reports/<config>/` | `REPORT.md`, `table14_…` … `table24_…`,  `fig8…fig12` | optional |



## Repository layout

```
configs/      full.yaml (the thesis matrix), quick.yaml (smoke test), replicates/ (other seeds)
src/llmprune/ pure Python, runs without PyTorch:  selection, counting, config, status, metrics, report, replicates, io_utils
              PyTorch:  modeling, importance, pruning, data, evaluation, benchmark, lora, pipeline, selftest
scripts/      thin wrappers (run_matrix.py, make_report.py, run_selftest.py, analytic_checks.py)
tests/        unit tests; the PyTorch ones skip themselves when torch is missing
docs/         REPRODUCING.md, SETTINGS_NOT_IN_THESIS.md
recipes/ results/   outputs worth committing (adapters/ and cache/ are git-ignored)
```

## Design notes

**Recipes instead of checkpoints.** None of the three methods changes a surviving weight: heads are sliced,
layers deleted, magnitude zeroes weights below a stored threshold. A small JSON recipe plus the base model
therefore reproduces a pruned model *exactly*, and no multi-gigabyte pruned checkpoint is stored. To use a
pruned model in your own code:

```python
from llmprune.config import load_config, expand_conditions
from llmprune.pipeline import load_condition_model

cfg = load_config("configs/full.yaml")
cond = next(c for c in expand_conditions(cfg) if c.id == "7b-heads-30-lora")
model, tokenizer = load_condition_model(cfg, cond)        # base model + recipe + LoRA adapter
```

(A head-pruned model cannot be reloaded with a plain `from_pretrained`: `config.num_attention_heads` stays
32 while the width differs per layer.)

**Head pruning is structural.** `apply_head_pruning` replaces the Q/K/V/O `nn.Linear` modules by smaller
ones, so executed matrices, parameter counts and the KV cache really shrink (`shapes_before` /
`shapes_after` in every recipe). It is checked against the same model with the removed heads' W_O columns
zeroed. Head scores follow Algorithm 1: `|Σ f·∂L/∂f|` per batch (`taylor_batch_size`, default 1), f being
the head output before W_O. At least one head per layer is kept, and the recipe records whether that rule
was binding.

**Magnitude pruning uses an exact threshold.** A histogram of the 15-bit magnitude patterns of all bf16
weights gives the exact quantile without sorting or sampling 6.5 B values; the threshold is stored in the
recipe and replay is verified against the stored zero count. Embeddings, `lm_head` and norms are never
touched. Only bf16/fp16 weights are supported.

**Layer pruning.** Block Influence is computed in one pass over the calibration data. Equation 5 as printed
is the relative Frobenius change `‖H_l − H_{l−1}‖_F / ‖H_{l−1}‖_F` (default, `bi_variant: frobenius`);
ShortGPT's `1 − cos` is computed in the same pass and selectable with `bi_variant: cosine`. Deleted
blocks are removed and the rest renumbered, so the KV cache stays aligned.

**LoRA (Table 11).** r = 16, α = 2r, dropout 0.05 on q/k/v/o, Alpaca-GPT4 (52,002 examples), AdamW,
weight decay 0.01, learning rate 3e-4 with cosine decay, micro-batch 4 × accumulation 8 = 32, 3 epochs
(1,625 optimiser steps per epoch in the thesis). The base model stays frozen in bf16 and the adapters train in fp32.
A hand-written loop is used because the effective batch is then exactly 32 (not multiplied by the number
of visible GPUs) and because it verifies that base weights, hence the zeros of magnitude pruning, are
unchanged after training (`base_weights_unchanged` in `train_summary.json`).

**Evaluation.** WikiText-2 perplexity: test split joined with blank lines, window 2,048, stride 512, exact
token-level negative log-likelihood. MMLU: 5-shot, macro average over the 57 subjects recomputed from the
per-subject results (the harness' group value is stored next to it). GLUE: SST-2, MNLI, QQP, QNLI via
lm-evaluation-harness 0.4.2 on the development sets.

**Efficiency benchmark (Section III.F, now fully specified).** Prompt of 128 random token ids, 256 greedy
new tokens, batch 1, 10 warm-up and 100 timed runs, `cuda.synchronize` around each run. Throughput is
`batch × new tokens / seconds` of the whole `generate` call (prefill included). Memory is
`torch.cuda.memory_allocated` (tensors only, summed over the GPUs that hold weights) minus a baseline, and
the peak is `max_memory_allocated`; both are in decimal GB. It is measured for `single` (one GPU) and
`auto` (`device_map="auto"`, the thesis setting).

\
## Statistics over seeds

The thesis uses one seed (42). Replicates are ordinary runs with other seeds and their own output folders
(`configs/replicates/seed43.yaml`, `seed44.yaml`):

```bash
llmprune run --config configs/replicates/seed43.yaml ...      # as above
llmprune aggregate --configs configs/full.yaml configs/replicates/seed43.yaml configs/replicates/seed44.yaml \
      --compare 7b-heads-30 7b-heads-30-lora --metric ppl      # mean, SD, 95 % CI; paired t-test, Cohen's d_z
```

Magnitude pruning without recovery is deterministic, so its spread is exactly zero. With only a few seeds
the confidence intervals are wide.

## Compute

Rough, unmeasured estimates — the first real run's logs (`elapsed_s` in `train_log.jsonl`) replace them:
one LoRA run on 7B is a few hours on one A100; MMLU 5-shot on 7B takes about an hour; perplexity minutes;
recipes minutes. The full matrix is therefore on the order of one to two days on four A100-80GB GPUs and
about a week on one. Disk: 13.5 GB (7B) + 26 GB (13B) of base models in the Hugging Face cache, plus adapters.

## Troubleshooting

* **`403` / gated repo:** accept the licence at `huggingface.co/meta-llama/Llama-2-7b-hf`, then `huggingface-cli login`. A local directory also works as `name_or_path`.
* **C4 cannot be downloaded:** fetch `en/c4-train.00000-of-01024.json.gz` from `huggingface.co/datasets/allenai/c4` and set `calibration.c4_local_path`.
* **`numpy.core.multiarray failed to import`:** NumPy 2 with PyTorch 2.2; `pip install numpy==1.26.4`.
* **Out of memory in LoRA:** lower `lora.micro_batch_size` and raise `lora.grad_accum` by the same factor (effective batch stays 32).
* **`lm_eval` complains about a model argument:** use the pinned `lm_eval==0.4.2`.
* **Windows:** shard locking needs POSIX (`fcntl`) and is disabled there; run one process at a time or use WSL/Linux.

## Not covered

Energy measurement (CodeCarbon), pruning + INT4 quantisation, grouped-query attention models and deployment tests on 12 GB GPUs are not implemented.

## Citation and licence

MIT licence (`LICENSE`). Cite the thesis and, if you use the code, this repository (`CITATION.cff`);
quote the commit hash, which is also stored in every result file.
