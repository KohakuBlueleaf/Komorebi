"""X280 ports on a card (KMR_CARD=1): twelve X280 cores answer every core through its
port, exactly."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
PORTS = (
    Path(__file__).resolve().parents[3] / "examples" / "10_x280_ports" / "x280_ports.py"
)


def test_twelve_x280_cores_answer_every_core_through_its_port_exactly():
    out = run_json(PORTS, "--rounds", 3, "--asks", 20)
    assert out["ok"] and out["exact"] and out["x280_cores"] == 12, out
