"""The overlay builder's slot check (experiments/overlay_probe): a slot that overlaps
reserved L1 is refused before anything is loaded."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "experiments/overlay_probe/builder.py"
SPEC = importlib.util.spec_from_file_location("overlay_builder", PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
RESERVED = [("kernel buffer", 0, 0x1B480), ("control words", 0x17FF00, 0x180000)]


def test_free_slot_passes():
    builder.check_slot(0x167F00, 96 * 1024, RESERVED)


def test_overlapping_slot_is_refused():
    with pytest.raises(builder.OverlayError, match="overlaps control words"):
        builder.check_slot(0x17FEC0, 256, RESERVED)
