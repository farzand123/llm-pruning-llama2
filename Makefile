PY ?= python

.PHONY: install install-dev test selftest checks check plan-quick quick report-quick status-quick plan-full full report-full status-full

install:            ## GPU stack pinned as in thesis Table 10 (install a CUDA torch build first if needed)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

install-dev:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

test:               ## fast tests (pure Python part always, PyTorch part when torch is installed)
	$(PY) -m pytest -q

selftest:           ## tiny random LLaMA: verifies pruning, LoRA and evaluation code, no downloads
	$(PY) scripts/run_selftest.py

checks:             ## reproduce the thesis' analytical numbers (parameters, FLOPs, KV cache, LoRA sizes)
	$(PY) scripts/analytic_checks.py

check:              ## environment versus the thesis' pinned versions, GPUs, Hugging Face token
	$(PY) -m llmprune.cli check

plan-quick:
	$(PY) scripts/run_matrix.py --config configs/quick.yaml --dry-run

quick:              ## whole pipeline on a 160M model (minutes): run this before the multi-day matrix
	$(PY) scripts/run_matrix.py --config configs/quick.yaml
	$(PY) scripts/make_report.py --config configs/quick.yaml

report-quick:
	$(PY) scripts/make_report.py --config configs/quick.yaml

status-quick:
	$(PY) -m llmprune.cli status --config configs/quick.yaml

plan-full:
	$(PY) scripts/run_matrix.py --config configs/full.yaml --dry-run

full:               ## the thesis matrix (days on 4 x A100); resumable, see README for sharding
	$(PY) scripts/run_matrix.py --config configs/full.yaml
	$(PY) scripts/make_report.py --config configs/full.yaml

report-full:
	$(PY) scripts/make_report.py --config configs/full.yaml

status-full:
	$(PY) -m llmprune.cli status --config configs/full.yaml
