"""A give-up that leaves page lines behind: the lost words named, the stop's EXIT
left done when the cores it waits for had ended by their own give-up."""

from types import SimpleNamespace

import numpy as np

from komorebi import abi
from komorebi import state as device_state
from komorebi.errors import describe_core
from komorebi.run import Run

CONSTANTS = abi.headers().constants


def live_words():
    return [0] * (CONSTANTS["KM_LIVE_READ"] // 4)


def test_a_line_not_served_and_a_number_abandoned_are_named():
    words = live_words()
    words[CONSTANTS["kLvLost"] + 1] = 9  # the writer: command 9, page 4, 3 pages
    words[CONSTANTS["kLvLostAt"] + 1] = 4
    words[CONSTANTS["kLvLost"]] = 7  # the reader: command 7 took line 12, unwritten
    words[CONSTANTS["kLvLostAt"]] = 12
    words[CONSTANTS["kLvLostPages"]] = (1 << 15 | 1 << 12) | (0 << 12 | 3) << 16
    lost = device_state.lost(words)
    assert lost == [
        {"kernel": "reader", "sequence": 7, "island": 1, "abandoned": 12},
        {"kernel": "writer", "sequence": 9, "island": 0, "first": 4, "pages": 3},
    ]
    row = describe_core({"core": 2, "live": device_state.live(words)})
    assert "command 7 took line 12 of the link core toward island 1" in row
    assert "3 page(s) from page 4 on island 0 not served in time" in row
    assert "may hold wrong data" in row
    assert device_state.lost(live_words()) == []


def test_the_stop_names_the_cores_that_ended_by_their_own_give_up():
    assert device_state.code_name("error", CONSTANTS["KMD_ERROR_ENDED"]) == "ended"
    rows = np.zeros((3, CONSTANTS["kStateWords"]), np.uint32)
    rows[:, CONSTANTS["kStMagic"]] = CONSTANTS["kStateMagic"]
    rows[1, CONSTANTS["kStEnds"]] = (
        CONSTANTS["kEndGaveUp"] | CONSTANTS["kEndGaveUp"] << 8
    )
    rows[1, CONSTANTS["kStWriter"]] = (
        CONSTANTS["kGaveBadItem"] | CONSTANTS["kWhereData"] << 8
    )
    rows[1, CONSTANTS["kStWriterSequence"]] = 30
    fake = SimpleNamespace(h=abi.headers(), cores=3, _state=rows)
    text = Run._ended_text(fake, {"seen": 1})
    assert text.startswith("the stop's EXIT was left done: 1 core(s) had ended")
    assert "core 1 writer gave up (bad_item) in data" in text
    assert "core 0" not in text and "core 2" not in text


def test_a_clean_stop_and_an_ended_stop_read_through_the_finish():
    h = abi.headers()
    rows = np.zeros((1, CONSTANTS["kStateWords"]), np.uint32)
    rows[0, CONSTANTS["kStMagic"]] = CONSTANTS["kStateMagic"]
    rows[0, CONSTANTS["kStEnds"]] = CONSTANTS["kEndGaveUp"]
    fake = SimpleNamespace(h=h, cores=1, _state=rows)
    fake._ended_text = lambda result: Run._ended_text(fake, result)
    clean = {"state": h["KMD_STATE_DONE"], "error": 0}
    assert Run._finish_problems(fake, clean) == ""
    ended = {**clean, "error": h["KMD_ERROR_ENDED"], "seen": 1}
    assert Run._finish_problems(fake, ended).startswith("the stop's EXIT was left done")
