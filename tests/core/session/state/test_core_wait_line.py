"""core.wait's and km::copy_l1's give-up line (live word kLvLine, role << 16 | line):
state.live reads it, and a core's row says where its kernel gave up (no card)."""

import numpy as np

from komorebi import abi
from komorebi.errors import describe_core
from komorebi.session import state

H = abi.headers()


def words(line=0):
    out = np.zeros(H["KM_LIVE_READ"] // 4, np.uint32)
    out[H["kLvLine"]] = line
    return out


def test_the_live_words_give_the_kernel_and_line_of_the_give_up():
    assert state.live(words())["line"] is None
    line = state.live(words(H["kRoleWriter"] << 16 | 1057))["line"]
    assert line == {"kernel": "writer", "line": 1057}


def test_a_core_s_row_names_the_line_beside_that_kernel_s_give_up():
    live = state.live(words(H["kRoleWriter"] << 16 | 1057))
    gave = {"code": "limit", "where": "data", "slot": 0, "sequence": 9}
    row = describe_core({"core": 7, "writer": gave, "reader": None, "live": live})
    assert row == "core 7 writer gave up (limit) in data at line 1057"
    row = describe_core({"core": 7, "reader": gave, "writer": None, "live": live})
    assert row == "core 7 reader gave up (limit) in data"  # the line is the writer's
