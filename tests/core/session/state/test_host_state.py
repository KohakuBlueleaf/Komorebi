"""Decoding the result block, the state words and the trace; Program."""

import numpy as np
import pytest

from komorebi import abi
from komorebi.errors import KomorebiError, describe_core
from komorebi.program.description import Buffer, Program
from komorebi.session import state as device_state

CONSTANTS = abi.headers().constants


def test_result_block_joins_wide_counts_and_names_codes():
    words = [0] * CONSTANTS["KMD_R_WORDS"]
    words[CONSTANTS["KMD_R_STATE"]] = CONSTANTS["KMD_STATE_FAILED"]
    words[CONSTANTS["KMD_R_ERROR"]] = CONSTANTS["KMD_ERROR_DEADLINE"]
    words[CONSTANTS["KMD_R_WHERE"]] = CONSTANTS["KMD_WHERE_LOOP"]
    words[CONSTANTS["KMD_R_SLOT"]], words[CONSTANTS["KMD_R_SEQUENCE"]] = 3, 77
    words[CONSTANTS["KMD_R_LAST_LO"]], words[CONSTANTS["KMD_R_LAST_HI"]] = 5, 2
    result = device_state.result_block(words)
    assert (result["state_name"], result["error_name"], result["where_name"]) == (
        "failed",
        "deadline",
        "loop",
    )
    assert (
        result["slot"] == 3
        and result["sequence"] == 77
        and result["last"] == 5 | 2 << 32
    )
    assert "last_lo" not in result and "last_hi" not in result


def state_rows(cores=4):
    rows = np.zeros((cores, CONSTANTS["kStateWords"]), dtype=np.uint32)
    rows[:, CONSTANTS["kStMagic"]] = (
        CONSTANTS["kStateMagic"] | 0xBEEF << 8 | np.arange(cores)
    )
    return rows


def test_core_state_and_gave_up_words():
    rows = state_rows()
    rows[2, CONSTANTS["kStReader"]] = (
        CONSTANTS["kGaveStop"] | CONSTANTS["kWhereStall"] << 8 | 5 << 16
    )
    rows[2, CONSTANTS["kStReaderSequence"]] = 9
    rows[1, CONSTANTS["kStWrong"]] = 2 << 8
    rows[1, CONSTANTS["kStEnds"]] = CONSTANTS["kEndExit"] | CONSTANTS["kEndAbort"] << 16
    rows[3, CONSTANTS["kStCompute"]] = (
        CONSTANTS["kGaveStop"] | CONSTANTS["kWhereCInput"] << 8
    )
    rows[3, CONSTANTS["kStComputeSequence"]] = 6
    state_two = device_state.core_state(rows, 2)
    assert state_two["magic_ok"] and state_two["writer"] is None
    assert state_two["reader"] == {
        "code": "stop",
        "where": "stall",
        "slot": 5,
        "sequence": 9,
    }
    state_one = device_state.core_state(rows, 1)
    assert state_one["compute_wrong"] == [0, 2, 0] and state_one["tag"] == 0xBEEF
    assert (
        state_one["reader_end"],
        state_one["writer_end"],
        state_one["compute_end"],
    ) == (1, 0, 3)
    assert device_state.core_state(rows, 3)["compute"] == {
        "code": "stop",
        "where": "cinput",
        "slot": 0,
        "sequence": 6,
    }
    limit = CONSTANTS["kGaveLimit"] | CONSTANTS["kWhereComplete"] << 8
    assert device_state.gave_up(limit, 4)["code"] == "limit"
    bad = CONSTANTS["kGaveBadItem"] | CONSTANTS["kWhereData"] << 8
    assert device_state.gave_up(bad, 2)["code"] == "bad_item"
    broadcast = CONSTANTS["kGaveBadBroadcast"] | CONSTANTS["kWhereBroadcastBegin"] << 8
    assert (
        device_state.gave_up(broadcast, 3)["code"],
        device_state.gave_up(broadcast, 3)["where"],
    ) == (
        "bad_broadcast",
        "broadcastbegin",
    )
    assert device_state.gave_up(0, 0) is None


def test_not_reported_names_the_ranks_that_did_not_report():
    rows = state_rows(8)
    for core in (2, 3, 5):
        rows[core, CONSTANTS["kStReported"] + 1] = 40
    out = device_state.not_reported(rows, (2, 5), slot=1, sequence=40)
    assert [entry["core"] for entry in out] == [4, 6]


def test_state_words_from_padded_control_blocks():
    width = CONSTANTS["KM_CONTROL_BYTES"] // 4 + 16
    rows = np.zeros((3, width), dtype=np.int64)
    start = 4 + CONSTANTS["KM_CONTROL_STATE"] // 4
    rows[:, start] = [10, 11, 12]
    rows[1, start + 1] = -1  # read back as int32
    words = device_state.state_words(rows, control_offset=4)
    assert words.shape == (3, CONSTANTS["kStateWords"]) and words.dtype == np.uint32
    assert list(words[:, 0]) == [10, 11, 12] and words[1, 1] == 0xFFFFFFFF


def test_port_slots_broadcast_areas_and_the_port_a_wait_named():
    block = np.zeros(CONSTANTS["KM_CONTROL_BYTES"] // 4, np.uint32)
    port = (CONSTANTS["KM_CONTROL_PORT"] + 3 * CONSTANTS["KM_PORT_STRIDE"]) // 4
    (
        block[port + CONSTANTS["KM_PORT_COUNT"]],
        block[port + CONSTANTS["KM_PORT_WANT"]],
    ) = (7, 9)
    broadcast = (
        CONSTANTS["KM_CONTROL_BROADCAST"] + CONSTANTS["KM_BROADCAST_BYTES"]
    ) // 4  # broadcast 1
    valid = broadcast + CONSTANTS["KM_BROADCAST_VALID"] // 4
    block[valid], block[valid + CONSTANTS["KM_BROADCAST_WORD"]] = 5 << 16 | 4, 77
    gave = broadcast + CONSTANTS["KM_BROADCAST_GAVE"] // 4
    block[gave : gave + 3] = [CONSTANTS["kWhereBroadcastValid"] | 1 << 8, 2, 5]
    stats = broadcast + CONSTANTS["KM_BROADCAST_STATS"] // 4
    block[stats : stats + 3] = [4, 10, 1]
    rank_2 = broadcast + CONSTANTS["KM_BROADCAST_DONE"] // 4 + 4 * 2
    block[rank_2] = 5 << 16 | 3
    block[rank_2 + CONSTANTS["KM_BROADCAST_ARRIVED"]] = 5 << 16 | 4
    ports = device_state.ports(block)
    assert len(ports) == CONSTANTS["KM_PORTS"] and ports[3] == {"count": 7, "want": 9}
    broadcasts = device_state.broadcasts(block)
    assert broadcasts[0]["gave_up"] is None and broadcasts[0]["blocks"] == 0
    assert (
        broadcasts[1]["valid"] == 5 << 16 | 4
        and broadcasts[1]["done"][1] == 5 << 16 | 3
    )
    assert broadcasts[1]["word"] == 77 and broadcasts[1]["arrived"][1] == 5 << 16 | 4
    assert broadcasts[1]["gave_up"] == {
        "where": "broadcastvalid",
        "broadcast": 1,
        "block": 2,
        "sequence": 5,
    }
    assert (broadcasts[1]["blocks"], broadcasts[1]["wait_cycles"]) == (4, 10 | 1 << 32)
    word = CONSTANTS["kGaveLimit"] | CONSTANTS["kWherePort"] << 8 | 6 << 16 | 3 << 24
    assert device_state.gave_up(word, 11) == {
        "code": "limit",
        "where": "port",
        "slot": 6,
        "sequence": 11,
        "port": 3,
    }


def test_trace_phases_and_overlap():
    rows = np.zeros((6, 8, 4), dtype=np.uint32)
    # handle 1: sequence 1 on cores 0..2; handle 2: sequence 2 on cores 3..5
    rows[0:3, 0] = [[1000, 2000, 2700, 5], [1100, 2100, 2600, 5], [1200, 2400, 2650, 5]]
    rows[3:6, 1] = [[2500, 2900, 3000, 1]] * 3
    trace = device_state.Trace(
        rows, {1: 1, 2: 2, 3: 99}, {1: (0, 3), 2: (3, 3), 3: (0, 1)}, 100
    )
    phases = trace.phases(1)
    assert (
        phases["count"] == 3
        and phases["last_seen"] == 2.0
        and phases["last_done"] == 14.0
    )
    assert phases["leader_sent"] == 17.0 and phases["push_after_last_done"] == 3.0
    assert trace.lead(1, 2) == 2.0  # handle 2 taken 2 us before handle 1's push
    assert trace.lead(2, 1) == 20.0
    assert trace.phases(4) is None and trace.command(4) is None  # no sequence
    wrapped = rows.copy()
    wrapped[0:3, 0, :3] += np.uint32(0xFFFFF000)
    assert device_state.Trace(wrapped, {1: 1}, {1: (0, 3)}, 100).phases(1) == phases


def test_a_command_past_the_trace_slots_is_named_with_what_took_them():
    rows = np.zeros((2, 4, 4), dtype=np.uint32)
    # handles 1..3 the run's, 4 and 5 the runtime's own, one command of another
    # source (sequence 6), then handle 6 at sequence 7: past the 4 slots
    seqs = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7}
    groups = {handle: (0, 2) for handle in seqs}
    own = {
        4: "the links' counts for crossing_line",
        5: "the links' counts for crossing_line",
    }
    trace = device_state.Trace(rows, seqs, groups, 100, own=own)
    assert trace.command(4).shape == (2, 4)
    with pytest.raises(KomorebiError) as error:
        trace.command(6)
    assert str(error.value) == (
        "handle 6: the trace holds 4 commands and this is command 7 of the cores' "
        "count (3 past it); before it: 3 of the run's commands, 2 of the runtime's "
        "own (the links' counts for crossing_line), 1 from other sources (a thread's "
        "commands): load with a larger trace="
    )
    with pytest.raises(KomorebiError, match="command 5 of the cores' count"):
        trace.phases(5)


def test_program_reads_handlers_and_checks_its_parts(tmp_path):
    for name in ("reader", "compute", "writer"):
        (tmp_path / f"{name}.cpp").write_text("")
    header = tmp_path / "p.h"
    header.write_text(
        "constexpr uint32_t kHandlerMap = 1;\nconstexpr uint32_t kCbIn = 0;\n"
    )
    program = Program(
        "p",
        kernels=tmp_path,
        buffers=[Buffer(0, "bfloat16", 2048, 4)],
        args=lambda core: ([core], [], [core * 2]),
        handlers=header,
    )
    assert program.handlers == {"map": 1} and program.handler("map") == 1
    assert program.sources()["compute"] == tmp_path / "compute.cpp"
    assert program.core_args(3) == ([3], [], [6]) and program.depth == 4
    with pytest.raises(FileNotFoundError):
        Program("q", kernels=tmp_path / "none").sources()
    with pytest.raises(ValueError, match="handler"):
        Program("q", kernels=tmp_path, handlers={"bad": 240})
    with pytest.raises(ValueError, match="depth"):
        Program("q", kernels=tmp_path, depth=65)
    with pytest.raises(ValueError, match="compile_args"):
        Program("q", kernels=tmp_path, compile_args={"feeder": []})


def test_dynamic_words_and_a_named_dynamic_give_up():
    block = np.zeros(CONSTANTS["KM_CONTROL_BYTES"] // 4, np.uint32)
    messages = CONSTANTS["KM_CONTROL_MESSAGES"] // 4
    for name, value in (("IN_TAIL", 40), ("IN_HEAD", 33), ("SENT", 9), ("QUOTA", 7)):
        block[messages + CONSTANTS[f"KM_MESSAGES_{name}"] // 4] = value
    block[messages + CONSTANTS["KM_MESSAGES_CONSUMED"] // 4] = 8
    assert device_state.dynamic(block)["gave_up"] is None
    gave = messages + CONSTANTS["KM_MESSAGES_GAVE"] // 4
    block[gave : gave + 4] = [CONSTANTS["kWhereMessagesLost"], 21, 3, 17]
    assert device_state.dynamic(block) == {
        "head": 33,
        "tail": 40,
        "sent": 9,
        "consumed": 8,
        "quota": 7,
        "gave_up": {"where": "lost", "position": 21, "run": 3, "seen": 17},
        "counter": None,
    }
    ready = (CONSTANTS["KM_CONTROL_TAILS"] + CONSTANTS["KM_TAILS_READY"]) // 4
    block[ready : ready + 4] = [3, 1, 59, 0]
    assert device_state.dynamic(block)["counter"] == {"run": 3, "serves": 59}
    codes = [
        CONSTANTS[f"kWhereMessages{where}"] for where in device_state.MESSAGES_WHERES
    ]
    # after the runtime's 1 to 18; handler and load follow the overlays' 36 to 38, the
    # body the transfers' and barriers' 43 to 49, direct, quota and refused later codes
    assert codes == [*range(19, 36), 40, 41, 50, 77, 83, 91]


def test_where_codes_are_one_list_without_repeats():
    codes = {
        name: code for name, code in CONSTANTS.items() if name.startswith("kWhere")
    }
    assert len(set(codes.values())) == len(codes) and min(codes.values()) >= 1


def test_a_put_into_a_far_dram_refused_on_the_card_names_its_island_and_hazard():
    word = (
        CONSTANTS["kGaveRefused"] | CONSTANTS["kWhereFarDram"] << 8 | (2 | 1 << 8) << 16
    )
    gave = device_state.gave_up(word, 5)
    assert (gave["code"], gave["where"], gave["island"]) == ("refused", "fardram", 1)
    row = describe_core({"core": 3, "island": 0, "rank": 4, "writer": gave})
    assert "writer gave up (refused) in fardram: a put into island 1's tensor" in row
    assert "hazards.md#a-link-packet-for-a-far-wormhole-chips-dram" in row


def test_no_free_slot_is_told_apart_from_the_table():
    word = (
        CONSTANTS["kGaveOverlay"]
        | CONSTANTS["kWhereOverlaySlots"] << 8
        | (1 | 1 << 8) << 16
    )
    gave = device_state.gave_up(word, 5)
    assert (gave["code"], gave["where"], gave["overlay"], gave["slots"]) == (
        "overlay",
        "overlayslots",
        1,
        1,
    )
    row = describe_core({"core": 3, "reader": gave, "writer": None})
    assert row.endswith(
        "reader gave up (overlay) in overlayslots: no free slot for overlay 1: "
        "1 slot, all held"
    )
    found = CONSTANTS["kGaveOverlay"] | CONSTANTS["kWhereOverlayFind"] << 8 | 2 << 16
    gave = device_state.gave_up(found, 5)
    assert gave["where"] == "overlayfind" and "slots" not in gave
    assert "no free slot" not in describe_core({"core": 3, "reader": gave})
