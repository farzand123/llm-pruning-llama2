#!/usr/bin/env python
"""Run (or resume) the experiment matrix without installing the package.

    python scripts/run_matrix.py --config configs/quick.yaml --dry-run
    python scripts/run_matrix.py --config configs/full.yaml

Equivalent to ``llmprune run ...``; see ``llmprune run --help``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmprune.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
