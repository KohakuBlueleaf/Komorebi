"""SwiGLU on a card through the dispatcher (KMR_CARD=1): a short call with the
gate fused into the second product, and with the gate a command of its own, is within
the check's bound of float64 and every core's kernels end."""

from pathlib import Path

import pytest
from child import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[2] / "projects" / "dense" / "swiglu.py"


@pytest.mark.parametrize("gate", [[], ["--gate-pass"]])
def test_a_short_call_is_within_the_bound(gate):
    out = run_json(SCRIPT, "--shape", "sw256", "--calls", 2, *gate)
    assert out["plan"]["gate_pass"] == bool(gate)
    assert out["run"]["ok"] and out["run"]["gave_up"] == []
    assert out["check"]["ok"] and out["ok"]
