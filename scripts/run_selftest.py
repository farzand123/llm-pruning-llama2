#!/usr/bin/env python
"""Run the tiny-model self-test (needs torch + transformers + peft; no GPU, no downloads).

    python scripts/run_selftest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmprune.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["selftest"]))
