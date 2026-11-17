"""An ask's or a put line's waits each have a name in a core's live words, so a
give-up's row says which one the core sat in (no card)."""

import numpy as np

from komorebi import abi
from komorebi.errors import describe_core
from komorebi.session import state

H = abi.headers()
WAITS = ("Confirm", "Number", "Place", "Slot", "Sent", "Count")


def test_each_ask_wait_has_its_own_name():
    names = state.where_names()
    assert [names[H[f"kWhereAsk{w}"]] for w in WAITS] == [
        f"ask_{w.lower()}" for w in WAITS
    ]
    assert len({H[f"kWhereAsk{w}"] for w in WAITS} | {H["kWhereAsk"]}) == 7


def test_a_core_s_row_says_which_ask_wait_its_writer_is_in():
    words = np.zeros(H["KM_LIVE_READ"] // 4, np.uint32)
    words[H["kLvWhere"] + H["kRoleWriter"]] = H["kWhereAskConfirm"]
    words[H["kLvSequence"] + H["kRoleWriter"]] = 29
    row = describe_core(
        {"core": 7, "reader": None, "writer": None, "live": state.live(words)}
    )
    assert row == "core 7 writer waits in ask_confirm"
