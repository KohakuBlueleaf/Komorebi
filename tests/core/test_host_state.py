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
