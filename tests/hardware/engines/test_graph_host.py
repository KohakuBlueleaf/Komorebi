"""The graph example on a card through the dispatcher (KMR_CARD=1), in one device
open: a stalled graph on the X280 ends in its deadline error, then the graph run
host-fed and on the X280 is exact on both paths."""

from pathlib import Path

import pytest
from shared_card import run_json

pytestmark = pytest.mark.card
SCRIPT = Path(__file__).resolve().parents[3] / "examples" / "06_graph" / "graph.py"


def test_a_stalled_graph_ends_in_its_error_then_both_paths_are_exact():
    out = run_json(SCRIPT, "--stall", "--deadline", "0.2")  # fails on a non-zero exit
    assert out["stall"] == {
        "on": "x280",
        "status": "error",
        "error": [8, 0],  # deadline, at node 0
        "kernel_error": 4,  # the kernel's deadline error
    }
    want = {"first": (5, 5, 2761255), "again": (7, 10, 25159642)}  # the numpy replay
    for path in out["paths"].values():
        for name, (rounds, iteration, error_code) in want.items():
            got = path[name]
            assert (got["rounds"], got["it"], got["err"]) == (
                rounds,
                iteration,
                error_code,
            )
            assert got["state_wrong"] == 0
        assert path["kernel_commands"] == out["simulated"]["kernel_commands"] == 49
        assert path["core_commands_ok"] and path["max_busy"] == 2
    assert out["paths"]["host"]["branches_overlapping_rounds"] == "5/5"
    assert out["paths_agree"] and out["ok"]
