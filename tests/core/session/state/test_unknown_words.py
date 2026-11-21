"""A card word past the host's tables (a live line's role) is reported by its raw
value beside the give-up it can read, and nothing raises (no card)."""

import numpy as np

from komorebi import abi
from komorebi.errors import describe_core
from komorebi.session import state

H = abi.headers()


def live_words(line):
    out = np.zeros(H["KM_LIVE_READ"] // 4, np.uint32)
    out[H["kLvLine"]] = line
    return out


def test_a_live_line_of_an_unknown_role_gives_its_raw_role():
    role = len(state.ROLES) + 2
    line = state.live(live_words(role << 16 | 1057))["line"]
    assert line == {"kernel": role, "line": 1057}


def test_a_core_s_row_keeps_its_give_up_and_names_the_unknown_role():
    live = state.live(live_words(0x7FFF << 16 | 12))
    gave = {"code": "limit", "where": "data", "slot": 0, "sequence": 9}
    row = describe_core({"core": 7, "writer": gave, "reader": None, "live": live})
    assert row == (
        "core 7 writer gave up (limit) in data live line 12 of unknown role 32767"
    )


def test_an_inspect_with_an_unknown_role_reads_whole():
    head, record = H["KMD_INSPECT_HEAD"] // 4, H["KMD_INSPECT_RECORD"] // 4
    block = np.zeros(head + record, np.uint32)
    block[H["KMD_I_MARK"]] = H["KMD_INSPECT_MARK"]
    block[H["KMD_I_KIND"]] = H["KMD_INSPECT_TIMEOUT"]
    block[H["KMD_I_DONE"]] = 1
    core = block[head:]
    core[H["KMD_IR_CORE"]] = 3
    core[H["KMD_IR_FLAGS"]] = H["KMD_INSPECT_READ"]
    words = core[H["KMD_INSPECT_STATE"] // 4 :]
    words[H["kStMagic"]] = H["kStateMagic"] & 0xFF000000 | 3
    words[H["kStWriter"]] = H["kGaveLimit"] | H["kWhereData"] << 8
    core[H["KMD_INSPECT_LIVE"] // 4 + H["kLvLine"]] = 9 << 16 | 40
    said = state.inspect(block, H)["cores"][0]
    assert said["writer"]["code"] == "limit" and said["writer"]["where"] == "data"
    assert said["live"]["line"] == {"kernel": 9, "line": 40}
    assert describe_core(said) == (
        "core 3 writer gave up (limit) in data live line 40 of unknown role 9"
    )
