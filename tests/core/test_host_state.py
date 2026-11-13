"""Decoding the result block, the state words and the trace; Program."""

import numpy as np
import pytest

from komorebi import abi
from komorebi import state as st
from komorebi.program import Buffer, Program

K = abi.headers().constants


def test_result_block_joins_wide_counts_and_names_codes():
    words = [0] * K["KMD_R_WORDS"]
    words[K["KMD_R_STATE"]] = K["KMD_STATE_FAILED"]
    words[K["KMD_R_ERROR"]] = K["KMD_ERR_DEADLINE"]
    words[K["KMD_R_WHERE"]] = K["KMD_WHERE_LOOP"]
    words[K["KMD_R_SLOT"]], words[K["KMD_R_SEQ"]] = 3, 77
    words[K["KMD_R_LAST_LO"]], words[K["KMD_R_LAST_HI"]] = 5, 2
    r = st.result_block(words)
    assert (r["state_name"], r["error_name"], r["where_name"]) == (
        "failed",
        "deadline",
        "loop",
    )
    assert r["slot"] == 3 and r["seq"] == 77 and r["last"] == 5 | 2 << 32
    assert "last_lo" not in r and "last_hi" not in r


def state_rows(cores=4):
    rows = np.zeros((cores, K["kStateWords"]), dtype=np.uint32)
    rows[:, K["kStMagic"]] = K["kStateMagic"] | 0xBEEF << 8 | np.arange(cores)
    return rows


def test_core_state_and_gave_up_words():
    rows = state_rows()
    rows[2, K["kStReader"]] = K["kGaveStop"] | K["kWhereStall"] << 8 | 5 << 16
    rows[2, K["kStReaderSeq"]] = 9
    rows[1, K["kStWrong"]] = 2 << 8
    rows[1, K["kStEnds"]] = K["kEndExit"] | K["kEndAbort"] << 16
    rows[3, K["kStCompute"]] = K["kGaveStop"] | K["kWhereCInput"] << 8
    rows[3, K["kStComputeSeq"]] = 6
    c = st.core_state(rows, 2)
    assert c["magic_ok"] and c["writer"] is None
    assert c["reader"] == {"code": "stop", "where": "stall", "slot": 5, "seq": 9}
    c1 = st.core_state(rows, 1)
    assert c1["compute_wrong"] == [0, 2, 0] and c1["tag"] == 0xBEEF
    assert (c1["reader_end"], c1["writer_end"], c1["compute_end"]) == (1, 0, 3)
    assert st.core_state(rows, 3)["compute"] == {
        "code": "stop",
        "where": "cinput",
        "slot": 0,
        "seq": 6,
    }
    limit = K["kGaveLimit"] | K["kWhereComplete"] << 8
    assert st.gave_up(limit, 4)["code"] == "limit"
    bad = K["kGaveBadItem"] | K["kWhereData"] << 8
    assert st.gave_up(bad, 2)["code"] == "bad_item"
    lane = K["kGaveBadLane"] | K["kWhereLaneBegin"] << 8
    assert (st.gave_up(lane, 3)["code"], st.gave_up(lane, 3)["where"]) == (
        "bad_lane",
        "lanebegin",
    )
    assert st.gave_up(0, 0) is None


def test_not_reported_names_the_members_that_did_not_report():
    rows = state_rows(8)
    for core in (2, 3, 5):
        rows[core, K["kStReported"] + 1] = 40
    out = st.not_reported(rows, (2, 5), slot=1, seq=40)
    assert [c["core"] for c in out] == [4, 6]


def test_state_words_from_padded_control_blocks():
    width = K["KM_CTL_BYTES"] // 4 + 16
    rows = np.zeros((3, width), dtype=np.int64)
    start = 4 + K["KM_CTL_STATE"] // 4
    rows[:, start] = [10, 11, 12]
    rows[1, start + 1] = -1  # read back as int32
    words = st.state_words(rows, ctl_offset=4)
    assert words.shape == (3, K["kStateWords"]) and words.dtype == np.uint32
    assert list(words[:, 0]) == [10, 11, 12] and words[1, 1] == 0xFFFFFFFF


def test_port_slots_lane_areas_and_the_port_a_wait_named():
    block = np.zeros(K["KM_CTL_BYTES"] // 4, np.uint32)
    port = (K["KM_CTL_PORT"] + 3 * K["KM_PORT_STRIDE"]) // 4
    block[port + K["KM_PORT_COUNT"]], block[port + K["KM_PORT_WANT"]] = 7, 9
    lane = (K["KM_CTL_LANE"] + K["KM_LANE_BYTES"]) // 4  # lane 1
    block[lane + K["KM_LANE_VALID"] // 4] = 5 << 16 | 4
    gave = lane + K["KM_LANE_GAVE"] // 4
    block[gave : gave + 3] = [K["kWhereLaneValid"] | 1 << 8, 2, 5]
    stats = lane + K["KM_LANE_STATS"] // 4
    block[stats : stats + 3] = [4, 10, 1]
    block[lane + K["KM_LANE_DONE"] // 4 + 4 * 2] = 5 << 16 | 3  # rank 2
    ports = st.ports(block)
    assert len(ports) == K["KM_PORTS"] and ports[3] == {"count": 7, "want": 9}
    lanes = st.lanes(block)
    assert lanes[0]["gave_up"] is None and lanes[0]["blocks"] == 0
    assert lanes[1]["valid"] == 5 << 16 | 4 and lanes[1]["done"][1] == 5 << 16 | 3
    assert lanes[1]["gave_up"] == {
        "where": "lanevalid",
        "lane": 1,
        "block": 2,
        "seq": 5,
    }
    assert (lanes[1]["blocks"], lanes[1]["wait_cycles"]) == (4, 10 | 1 << 32)
    word = K["kGaveLimit"] | K["kWherePort"] << 8 | 6 << 16 | 3 << 24
    assert st.gave_up(word, 11) == {
        "code": "limit",
        "where": "port",
        "slot": 6,
        "seq": 11,
        "port": 3,
    }


def test_trace_phases_and_overlap():
    rows = np.zeros((6, 8, 4), dtype=np.uint32)
    # ticket 1: seq 1 on cores 0..2; ticket 2: seq 2 on cores 3..5
    rows[0:3, 0] = [[1000, 2000, 2700, 5], [1100, 2100, 2600, 5], [1200, 2400, 2650, 5]]
    rows[3:6, 1] = [[2500, 2900, 3000, 1]] * 3
    trace = st.Trace(rows, {1: 1, 2: 2, 3: 99}, {1: (0, 3), 2: (3, 3), 3: (0, 1)}, 100)
    p = trace.phases(1)
    assert p["count"] == 3 and p["last_seen"] == 2.0 and p["last_done"] == 14.0
    assert p["leader_sent"] == 17.0 and p["push_after_last_done"] == 3.0
    assert trace.lead(1, 2) == 2.0  # ticket 2 taken 2 us before ticket 1's push
    assert trace.lead(2, 1) == 20.0
    assert trace.phases(3) is None and trace.record(3) is None
    wrapped = rows.copy()
    wrapped[0:3, 0, :3] += np.uint32(0xFFFFF000)
    assert st.Trace(wrapped, {1: 1}, {1: (0, 3)}, 100).phases(1) == p


def test_program_reads_segments_and_checks_its_parts(tmp_path):
    for name in ("reader", "compute", "writer"):
        (tmp_path / f"{name}.cpp").write_text("")
    header = tmp_path / "p.h"
    header.write_text(
        "constexpr uint32_t kSegMap = 1;\nconstexpr uint32_t kCbIn = 0;\n"
    )
    program = Program(
        "p",
        kernels=tmp_path,
        buffers=[Buffer(0, "bfloat16", 2048, 4)],
        args=lambda core: ([core], [], [core * 2]),
        segments=header,
    )
    assert program.segments == {"map": 1} and program.seg("map") == 1
    assert program.sources()["compute"] == tmp_path / "compute.cpp"
    assert program.core_args(3) == ([3], [], [6]) and program.depth == 4
    with pytest.raises(FileNotFoundError):
        Program("q", kernels=tmp_path / "none").sources()
    with pytest.raises(ValueError, match="segment"):
        Program("q", kernels=tmp_path, segments={"bad": 240})
    with pytest.raises(ValueError, match="depth"):
        Program("q", kernels=tmp_path, depth=17)
    with pytest.raises(ValueError, match="compile_args"):
        Program("q", kernels=tmp_path, compile_args={"feeder": []})


def test_dynamic_words_and_a_named_dynamic_give_up():
    block = np.zeros(K["KM_CTL_BYTES"] // 4, np.uint32)
    dyn = K["KM_CTL_DYN"] // 4
    for name, value in (("IN_TAIL", 40), ("IN_HEAD", 33), ("SENT", 9), ("QUOTA", 7)):
        block[dyn + K[f"KM_DYN_{name}"] // 4] = value
    block[dyn + K["KM_DYN_CONSUMED"] // 4] = 8
    assert st.dynamic(block)["gave_up"] is None
    gave = dyn + K["KM_DYN_GAVE"] // 4
    block[gave : gave + 4] = [K["kWhereDynLost"], 21, 3, 17]
    assert st.dynamic(block) == {
        "head": 33,
        "tail": 40,
        "sent": 9,
        "consumed": 8,
        "quota": 7,
        "gave_up": {"where": "lost", "position": 21, "run": 3, "seen": 17},
    }
    codes = [K[f"kWhereDyn{k}"] for k in st.DYN_WHERES]
    assert codes == list(range(19, 19 + len(codes)))  # after the runtime's 1 to 18


def test_where_codes_are_one_list_without_repeats():
    codes = {k: v for k, v in K.items() if k.startswith("kWhere")}
    assert len(set(codes.values())) == len(codes) and min(codes.values()) >= 1
