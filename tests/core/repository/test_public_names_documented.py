"""Every public name is in the reference: each name `komorebi` exports, and each tile
function a class calls (`km.exp`, `km.relu`, ...), appears in a page of
docs/reference, so a name added without its page fails here."""

import re
from pathlib import Path

import komorebi
from komorebi.dsl.front.names import NAMES

REFERENCE = Path(__file__).resolve().parents[3] / "docs/reference"


def reference_text():
    return "\n".join(page.read_text() for page in sorted(REFERENCE.glob("*.md")))


def test_every_exported_name_is_in_the_reference():
    text = reference_text()
    missing = [
        name
        for name in komorebi.__all__
        if not re.search(rf"(\bkm\.|\bkomorebi\.|`){re.escape(name)}\b", text)
    ]
    assert not missing, f"not in docs/reference: {missing}"


def test_every_tile_function_is_in_the_reference():
    text = reference_text()
    functions = [
        name.spelling for key, name in NAMES.items() if key.startswith("math_")
    ]
    missing = [
        name
        for name in functions
        if not re.search(rf"\b(km|trace)\.{re.escape(name)}\b", text)
    ]
    assert functions and not missing, f"not in docs/reference: {missing}"
