#!/usr/bin/env python
"""Regenerate the thesis tables (14-24), figures (8-12) and the comparison with the printed values.

    python scripts/make_report.py --config configs/full.yaml [--out reports/full]

Equivalent to ``llmprune report ...``. Needs no GPU stack; partial results give a partial report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmprune.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))
