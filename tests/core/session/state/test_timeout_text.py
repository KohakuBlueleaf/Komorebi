"""A CommandTimeout's list of not-reported cores names the cores that waited first, and
counts, not lists, a core that shows only its reader's normal stop."""

import numpy as np

from komorebi import abi
from komorebi.errors import CommandTimeout, _only_normal_stop
from komorebi.session import state as device_state

H = abi.headers()
COLUMNS, CORES = 12, 119  # a Tensix dispatcher beside the last core of a 12 x 10 grid


def stopped_state(reporting):
    """Every core's state words after a stop: the readers' normal stop in next; the
    cores in `reporting` reported sequence 1 in slot 0."""
    words = np.zeros((CORES, H["kStMagic"] + 16), np.uint32)
    for core in range(CORES):
        words[core, H["kStMagic"]] = H["kStateMagic"] | core
        words[core, H["kStReader"]] = H["kGaveStop"] | H["kWhereNext"] << 8
        if core in reporting:
            words[core, H["kStReported"]] = 1
    return words


def test_the_cores_that_waited_come_first_and_a_normal_stop_is_counted():
    leaders = [row * COLUMNS for row in range(9)]  # column 0, rows 0 to 8
    silent = [row * COLUMNS + 11 for row in range(9)]  # column 11: in no broadcast
    words = stopped_state(set(range(CORES)) - set(leaders) - set(silent))
    for core in leaders:  # its writer waited in complete when the stop came
        words[core, H["kStWriter"]] = H["kGaveStop"] | H["kWhereComplete"] << 8
        words[core, H["kStWriterSequence"]] = 1
    cores = device_state.not_reported(words, (0, CORES), 0, 1, H)
    assert [core["core"] for core in cores] == sorted(leaders + silent)
    text = str(CommandTimeout(1, 0, 1, (0, CORES), cores))
    listed = [
        f"core {core} reader gave up (stop) in next writer gave up" for core in leaders
    ]
    assert all(line in text for line in listed)
    assert text.index(listed[0]) < text.index(listed[-1])
    assert not any(f"core {core} " in text for core in silent)
    assert (
        "9 more cores did not report and show only their reader's normal stop" in text
    )
    assert [core["core"] for core in cores] == sorted(leaders + silent)  # .cores whole


def test_a_list_of_only_normal_stops_is_listed_as_before():
    words = stopped_state(set())
    cores = device_state.not_reported(words, (0, 3), 0, 1, H)
    assert all(_only_normal_stop(core) for core in cores)
    text = str(CommandTimeout(1, 0, 1, (0, 3), cores))
    assert "core 0 reader gave up (stop) in next; core 1 reader" in text
    assert "not listed" not in text
