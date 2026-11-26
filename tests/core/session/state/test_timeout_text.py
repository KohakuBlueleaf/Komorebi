"""A CommandTimeout's list of not-reported cores has two groups by index: the cores
that waited, with where, then the cores that show only their reader's normal stop."""

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


def test_both_groups_are_listed_by_index_under_their_labels():
    leaders = [row * COLUMNS for row in range(9)]  # column 0, rows 0 to 8
    silent = [row * COLUMNS + 11 for row in range(9)]  # column 11: in no broadcast
    words = stopped_state(set(range(CORES)) - set(leaders) - set(silent))
    for core in leaders:  # its writer waited in complete when the stop came
        words[core, H["kStWriter"]] = H["kGaveStop"] | H["kWhereComplete"] << 8
        words[core, H["kStWriterSequence"]] = 1
    cores = device_state.not_reported(words, (0, CORES), 0, 1, H)
    assert [core["core"] for core in cores] == sorted(leaders + silent)
    text = str(CommandTimeout(1, 0, 1, (0, CORES), cores))
    waited = ", ".join(map(str, leaders))
    quiet = ", ".join(map(str, silent))
    assert (
        f"cores: waited: cores {waited}: reader gave up (stop) in next writer gave up "
        f"(stop) in complete; did not report and show only their reader's normal stop "
        f"in next: cores {quiet}"
    ) in text
    assert [core["core"] for core in cores] == sorted(leaders + silent)  # .cores whole


def test_a_long_group_lists_32_cores_then_a_count():
    cores = [{"core": core, "reader": None, "writer": None} for core in range(100, 140)]
    cores[0]["writer"] = {"code": "limit", "where": "complete", "sequence": 1}
    cores += [
        {"core": core, "reader": {"code": "stop", "where": "next"}}
        for core in range(40)
    ]
    text = str(CommandTimeout(1, 0, 1, (0, 140), cores))
    assert "waited: cores 100: writer gave up (limit) in complete" in text
    assert "did not report, no give-up recorded: cores 101, 102" in text
    quiet = text.split("normal stop in next: cores ")[1]
    assert quiet == ", ".join(map(str, range(32))) + " and 8 more"


def test_a_list_of_only_normal_stops_is_listed_as_before():
    words = stopped_state(set())
    cores = device_state.not_reported(words, (0, 3), 0, 1, H)
    assert all(_only_normal_stop(core) for core in cores)
    text = str(CommandTimeout(1, 0, 1, (0, 3), cores))
    assert "core 0 reader gave up (stop) in next; core 1 reader" in text
    assert "waited:" not in text


def test_a_core_without_this_runs_magic_is_named_apart_and_counted_among_the_reset():
    head, record = H["KMD_INSPECT_HEAD"] // 4, H["KMD_INSPECT_RECORD"] // 4
    group = [0, 12, 24, 36, 60]
    block = np.zeros(head + record * len(group), np.uint32)
    block[H["KMD_I_MARK"]], block[H["KMD_I_KIND"]] = (
        H["KMD_INSPECT_MARK"],
        H["KMD_INSPECT_TIMEOUT"],
    )
    block[H["KMD_I_COUNT"]] = block[H["KMD_I_DONE"]] = len(group)
    block[H["KMD_I_SEQUENCE"]] = 5
    state_at, live_at = H["KMD_INSPECT_STATE"] // 4, H["KMD_INSPECT_LIVE"] // 4
    for index, core in enumerate(group):
        line = block[head + index * record :]
        line[H["KMD_IR_CORE"]] = core
        line[H["KMD_IR_FLAGS"]] = H["KMD_INSPECT_READ"]
        if core == 36:  # zeros: no magic, no tag, no live word
            line[H["KMD_IR_FLAGS"]] |= H["KMD_INSPECT_RESET"]
            continue
        line[state_at + H["kStMagic"]] = H["kStateMagic"] | 0x6543 << 8 | core
        sequences = live_at + H["KM_LIVE_SEQUENCE"]
        line[sequences : sequences + H["KM_LIVE_ROLES"]] = 5
        if core == 60:  # its reader's word missing: the stalled one
            line[sequences] = 0
            line[H["KMD_IR_FLAGS"]] |= H["KMD_INSPECT_RESET"]
    cores = device_state.inspect(block, H, tag=0x6543)["cores"]
    assert [core["core"] for core in cores if core["not_there"]] == [36]
    reset = [core["core"] for core in cores if core["reset"]]
    text = str(CommandTimeout(6, 0, 5, (0, 109), cores, given_up=True, recovered=reset))
    assert (
        "core 36: its control area is not there (no kernel of this run is running on "
        "it): reset and loaded again"
    ) in text
    assert "recovered (reset and loaded again): cores [36, 60]" in text
    other = device_state.inspect(block, H, tag=0x6545)["cores"]  # another run's tag
    assert all(core["not_there"] for core in other)


def test_a_reset_cores_gave_up_word_is_named_with_where_it_is_kept():
    head, record = H["KMD_INSPECT_HEAD"] // 4, H["KMD_INSPECT_RECORD"] // 4
    block = np.zeros(head + record * 2, np.uint32)
    block[H["KMD_I_MARK"]], block[H["KMD_I_KIND"]] = (
        H["KMD_INSPECT_MARK"],
        H["KMD_INSPECT_TIMEOUT"],
    )
    block[H["KMD_I_COUNT"]] = block[H["KMD_I_DONE"]] = 2
    block[H["KMD_I_SEQUENCE"]] = 1
    state_at = H["KMD_INSPECT_STATE"] // 4
    for core in range(2):
        line = block[head + core * record :]
        line[H["KMD_IR_CORE"]] = core
        line[H["KMD_IR_FLAGS"]] = H["KMD_INSPECT_READ"]
        line[state_at + H["kStMagic"]] = H["kStateMagic"] | 0x6543 << 8 | core
    # core 1's reader gave up by its own limit in port, then the deadline reset it
    line = block[head + record :]
    line[state_at + H["kStReader"]] = H["kGaveLimit"] | H["kWherePort"] << 8
    line[state_at + H["kStReaderSequence"]] = 1
    line[H["KMD_IR_FLAGS"]] |= H["KMD_INSPECT_RESET"]
    cores = device_state.inspect(block, H, tag=0x6543)["cores"]
    reset = [core["core"] for core in cores if core["reset"]]
    text = str(CommandTimeout(2, 0, 1, (0, 2), cores, given_up=True, recovered=reset))
    assert "core 1 reader gave up (limit) in port" in text
    assert text.endswith(
        "recovered (reset and loaded again): cores [1]; the gave-up words above were "
        "read at the deadline, before the reset: kept in this error's cores, "
        "run.given_up(handle) and run.recoveries, no longer in run.state() after the "
        "stop"
    )
    kept = str(CommandTimeout(2, 0, 1, (0, 2), cores, given_up=True))
    assert "gave-up words" not in kept  # no core reset: run.state() still has them
