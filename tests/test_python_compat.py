"""The thesis environment is Python 3.10: reject syntax newer than that, whichever Python runs the tests."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_parses_as_python_310(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
