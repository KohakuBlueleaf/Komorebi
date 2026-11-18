"""tests/core/dsl: each folder holding a helper module (`*_fixture.py`) on sys.path, so
a test of another folder imports it by name (pytest puts only the test file's own
folder there)."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in sorted({path.parent for path in HERE.rglob("*_fixture.py")}):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
