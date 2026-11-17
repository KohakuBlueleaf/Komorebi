"""tests/core/dsl: this folder and each of its test folders on sys.path, so a test
imports chain_fixture or another folder's test module by name (pytest puts only the
test file's own folder there)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in [HERE, *sorted(HERE.rglob("*"))]:
    tests = folder == HERE or any(folder.glob("test_*.py"))
    if folder.is_dir() and tests and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
