# Reproducing the thesis experiments, step by step

This is the long version of the README's quick start. Commands are run from the repository root.

## 0. What you need

* Linux with NVIDIA GPUs. The thesis used 4 × A100-SXM4 80 GB. A single 80 GB GPU should be enough for
  everything except the `auto` efficiency setting (it needs at least two GPUs and is skipped otherwise);
  memory was not measured for this repository, and smaller GPUs were not considered. bf16 needs Ampere or newer.
* A Hugging Face account that has accepted the LLaMA-2 licence, with a token (`huggingface-cli login`).
* Disk: about 40 GB for the base models, a few GB for adapters, little else.
* Internet access for the models, C4, WikiText-2, Alpaca-GPT4 and the lm-evaluation-harness tasks. Offline
  clusters: download once, point `models.*.name_or_path` at local directories and set `calibration.c4_local_path`.

## 1. Environment

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121   # pick the CUDA build of your driver
make install-dev
make check          # compares installed versions with thesis Table 10 and lists GPUs
```

`make check` exits non-zero if a pinned package is missing and prints `differs` for other versions; the
code is written for the pinned versions and not tested against others.

## 2. Verify the installation before spending GPU-days

```bash
make selftest   # 11 checks on a tiny random LLaMA: pruning == masking, layer renumbering and KV cache, exact
                # magnitude threshold, Taylor scores == gradient of an explicit gate, Block Influence, recipe
                # round trip, LoRA (parameter count, fp32 adapters, frozen base), sliding-window perplexity,
                # benchmark, GQA guard
make test       # unit tests (the PyTorch ones run when torch is installed)
make checks     # analytical numbers of the thesis: parameters, FLOPs (Eq. 13), KV cache, LoRA sizes (Table 24)
```

If `make selftest` fails, **stop and fix that first**; the long runs would produce wrong numbers silently.
If a check fails on the pinned versions, open an issue with the traceback.

## 3. Smoke test on a small model

```bash
make quick      # configs/quick.yaml: a 160M LLaMA, 16 calibration sequences, 10 LoRA steps, 16 evaluation examples
```

This runs every stage (recipe, LoRA, evaluation, benchmark, report) end to end in minutes and writes to
`recipes_quick/`, `adapters_quick/`, `results_quick/`, `reports/quick/`. The numbers are meaningless; what
matters is that every file appears (`llmprune status --config configs/quick.yaml`).

## 4. Settings the thesis leaves open

`configs/full.yaml` and `docs/SETTINGS_NOT_IN_THESIS.md` list every value the thesis does not state. The one
that changes results most is the **GLUE protocol** (`evaluation.glue.num_fewshot`, default 0); the choice is
stored in every `eval_glue.json`.

## 5. Baseline first

```bash
llmprune run --config configs/full.yaml --only 7b-base --stages eval
cat results/7b-base/eval_ppl.json     # wikitext2_ppl: expect roughly 5.1-5.5, the thesis prints 6.97
```

Perplexity is the first evaluation to finish (it is followed by GLUE and MMLU, about an hour); you can interrupt
the run once `eval_ppl.json` exists and resume later, because every result file is written when complete.

If the perplexity is far from that range, check the protocol (`evaluation.ppl`), the tokenizer and the dtype
before running anything else.

## 6. The matrix

```bash
llmprune plan --config configs/full.yaml                     # Table 17 as a list; add --include-disabled for No. 40

# four GPUs
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k nohup llmprune run --config configs/full.yaml \
      --stages recipe,lora,eval --shard $k/4 --continue-on-error > logs_shard$k.txt 2>&1 &
done

llmprune status --config configs/full.yaml --missing         # progress; also lists stale outputs
```

* The shards split the conditions round-robin; the expensive 13B ones are few. Re-run the same command to resume.
* `--continue-on-error` moves on after a failure and prints the failed ids at the end.
* After the shards finish, run the efficiency benchmark on idle GPUs, without `CUDA_VISIBLE_DEVICES`:
  `llmprune run --config configs/full.yaml --stages efficiency`. Do not run anything else on those GPUs meanwhile.
* `--force` recomputes the selected stages and everything built on them.
* One GPU only: `llmprune run --config configs/full.yaml` (add `--only` to work in pieces).

## 7. Report

```bash
llmprune report --config configs/full.yaml        # reports/full/
```

* `REPORT.md` — all tables with notes; `table14_…` … `table24_….csv`; figures `fig8…fig12`.
* `comparison_with_thesis.csv` — one row per printed number: reproduced value, printed value, difference.
  Read the large differences first (baseline perplexity, GLUE, speed-up, memory): see the README section "Differences from the printed thesis values".
* Cells whose files do not exist yet show `–`; the report never invents a number.

## 8. Replicates for the statistics

```bash
llmprune run --config configs/replicates/seed43.yaml --stages recipe,lora,eval --shard $k/4   # as above
llmprune run --config configs/replicates/seed44.yaml ...
llmprune aggregate --configs configs/full.yaml configs/replicates/seed43.yaml configs/replicates/seed44.yaml \
        --out reports/replicates --compare 7b-heads-30 7b-heads-30-lora --metric ppl
```

A replicate repeats the *whole* pipeline with another seed (new calibration sample, new LoRA data order and
initialisation) into separate folders. Restrict it with `--only` if you cannot afford everything, e.g. the
30 % and 50 % conditions. Efficiency benchmarks do not depend on the seed; their spread over the 100 runs is
`tokens_per_s_std`.

## 9. Archive and cite

```bash
git add recipes results reports configs docs src tests scripts README.md requirements*.txt
git commit -m "Thesis results"          # adapters/ and cache/ are ignored
git tag -a v1.0.0 -m "Thesis experiments"
```

* Every result and recipe stores the commit, library versions, hardware and timestamp; cite the tag and commit in the thesis.
* Publish the adapters (`adapters/<condition>/`) on the Hugging Face Hub or as a GitHub release asset, and
  archive the release (for example on Zenodo) to get a DOI.
* Keep `logs_shard*.txt` and `adapters/*/train_log.jsonl` with the release: they are the training and evaluation logs of the run.

## 10. Output files

| File | Keys that matter |
|---|---|
| `recipes/<model>-<method>-<pct>.json` | `method`, `pct`, `ratio`, `heads_keep` / `layers_removed` / `magnitude.pattern_threshold`, `head_selection`, `head_scores` / `bi_scores`, `params_before`, `params_after` (`structural`, `nonzero`, `weight_bytes`), `shapes_before`, `shapes_after`, `verification`, `calibration` (incl. token hash), `git`, `environment` |
| `results/<id>/eval_ppl.json` | `wikitext2_ppl`, `nll_per_token`, `n_tokens`, `protocol`, `provenance` |
| `results/<id>/eval_mmlu.json` | `macro_average_pct`, `harness_group_acc_pct`, `per_subject_acc_pct`, `protocol` |
| `results/<id>/eval_glue.json` | `acc_pct` per task, `average_pct`, `protocol` |
| `results/<id>/efficiency_<setting>.json` | `tokens_per_s_mean/std/median/min/max`, `weights_allocated_gb`, `peak_allocated_gb`, `devices`, `protocol`, `hardware` |
| `adapters/<id>/train_summary.json` | `trainable_params`, `steps`, `first_loss`, `final_loss`, `loss_curve`, `base_weights_unchanged`, `finished`, `provenance` |

## 11. When something goes wrong

* **A stage fails after a long time.** Rerun the same command; finished stages are skipped. Use `llmprune status`.
* **"head pruning does not reproduce the masked reference".** An indexing or version problem: run `make selftest`; do not disable `verify` to get past it.
* **"recipe zeroes N weights but M fell below the stored threshold".** The base weights differ from those the recipe was made with (other revision or dtype). Recompute the recipe with `--force` and everything built on it.
* **Different results on rerun with the same seed.** GPU kernels are not bit-deterministic across hardware; compare to replicate spread, not to the last digit.
