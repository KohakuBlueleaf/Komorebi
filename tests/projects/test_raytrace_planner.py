"""The X280 frame thread's table and the numbers plan.h shares with the kernels."""

import numpy as np
import pytest

from projects.raytrace import packets, planner

CONSTANTS = packets.K
P = planner.P


class Tensor:
    def __init__(self, address):
        self.address = address


def test_plan_h_agrees_with_the_kernels():
    assert planner.same_numbers() == {}
    assert P["RP_COST_BYTES"] == CONSTANTS["kCostBytes"]


def test_table_holds_commands_owners_and_profiles():
    directory = np.zeros((3, CONSTANTS["kDirBytes"] // 4), dtype=np.uint32)
    directory[:, CONSTANTS["kDirFirst"]] = [0, 5, 9]
    directory[:, CONSTANTS["kDirCount"]] = [5, 4, 2]
    entries = [(P["RP_KIND_FIRST"], 7, [1, 2]), (2, 5, list(range(10)))]
    profs = [[Tensor(100 + k) for k in range(2)] for _ in range(planner.P["RP_TRACES"])]
    t = planner.table(entries, directory, profs, 3)
    rec = t[: 2 * P["RP_REC_WORDS"]].reshape(2, -1)
    assert rec[0, :5].tolist() == [P["RP_KIND_FIRST"], 7, 3 << 16, 1, 2]
    assert rec[1, :3].tolist() == [2, 5, 3 << 16] and rec[1, 3:].tolist() == list(
        range(10)
    )
    assert t[P["RP_T_BASE"] : P["RP_T_BASE"] + 6].tolist() == [0, 5, 5, 4, 9, 2]
    assert t[P["RP_T_PROF"] + P["RP_CHUNKS"] + 1] == 101
    with pytest.raises(ValueError, match="table holds"):
        planner.table([entries[1]] * (P["RP_COMMANDS_MAX"] + 1), directory, profs, 3)


def test_entries_make_one_per_trace():
    steps = (
        ("gen", "kHandlerGen"),
        ("primary", "kHandlerTrace"),
        ("shadow", "kHandlerTrace"),
    )
    recs = [(9, [0, 0]), (1, [0] * 10), (5, [3] * 10), (5, [4] * 10), (5, [5] * 10)]
    out = planner.entries_of(recs, steps, [1, 2, 1], treelets=True)
    assert [e[0] for e in out] == [P["RP_KIND_FIRST"], 0, 1, 2]
    assert out[2][2][0] == 0 and out[2][2][4] == 0 and out[2][2][1] == 3
