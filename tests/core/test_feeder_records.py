"""The feeder's host helpers and the pure record logic of the device headers."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from komorebi.layout import share

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "feeder_records", ROOT / "tests" / "hardware" / "feeder" / "records.py"
)
rec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rec)
K = rec.K
GRID = (12, 10)


def test_record_fields_follow_the_abi_header():
    assert rec.RECORD_FIELDS == [
        "seq",
        "hdr",
        "team",
        "a0",
        "land1",
        "a1",
        "a2",
        "a3",
        "land2",
        "a4",
        "a5",
        "a6",
        "land3",
        "a7",
        "a8",
        "a9",
    ]


def test_constants_come_from_the_headers():
    assert K["KM_RING_SLOTS"] == 16 and K["KM_CTL_BYTES"] == 0x15C0
    assert K["KM_READY"] == 0xFFFFFFFF and K["KM_SEG_EXIT"] == 255
    assert K["KM_SEG_USER_MAX"] == 239 and "KM_F_ACK" not in K
    assert K["kStateWords"] * 4 <= K["KM_CTL_JOB"] - K["KM_CTL_STATE"]
    assert K["kFdArgs"] == 15 and K["kFeederMagic"] == 0xFEED0000


def test_start_up_arguments_are_in_header_order():
    assert rec.ARG_FIELDS[:3] == ["ctl", "me", "cores"]
    assert len(rec.ARG_FIELDS) == 19
    assert {"ready_base", "row", "workers"} <= set(rec.ARG_FIELDS)
    values = {name: i for i, name in enumerate(rec.ARG_FIELDS)}
    assert rec.start_args(values) == list(range(19))
    with pytest.raises(ValueError, match="missing"):
        rec.start_args({"ctl": 1})


def test_record_encoding():
    words = rec.record(7, 5, 60, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    w = dict(zip(rec.RECORD_FIELDS, words))
    assert w["hdr"] == 7  # flags zero: every record completes
    assert w["team"] == 5 | 60 << 16
    assert [w[f"a{i}"] for i in range(10)] == list(range(1, 11))
    assert w["seq"] == w["land1"] == w["land2"] == w["land3"] == 0
    with pytest.raises(ValueError):
        rec.record(1, 0, 1, list(range(11)))


def test_int32_round_trip_keeps_the_bits():
    rows = [[0xFFFFFFFF, 0x80000000, 5]]
    assert rec.as_uint32(rec.as_int32(rows)).tolist() == rows


@pytest.mark.parametrize("workers", [119, 120, 13, 1])
def test_groups_cover_every_worker_once(workers):
    grp, n_groups = rec.groups(GRID, workers)
    assert sorted(grp) == list(range(workers))
    leaders = {c for c, g in grp.items() if g[0] == c}
    assert n_groups == len(leaders) <= K["KM_RING_SLOTS"]
    for leader in leaders:
        members = [c for c, g in grp.items() if g[0] == leader and c != leader]
        assert grp[leader][2] == len(members)
        assert sorted(grp[c][1] for c in members) == list(range(len(members)))
        assert all(c % GRID[0] == leader % GRID[0] and c > leader for c in members)
    assert len({g[3] for c, g in grp.items() if g[0] == c}) == n_groups


def test_share_matches_the_layout_share():
    for n in (0, 1, 7, 64, 120):
        for count in (1, 3, 8, 60, 119):
            for i in range(count):
                start, length = rec.share(n, i, count)
                assert range(start, start + length) == share(range(n), i, count)


def test_random_teams_fit_the_workers():
    rng = np.random.default_rng(3)
    teams = rec.random_teams(rng, 500, 119)
    assert all(c >= 1 and f >= 0 and f + c <= 119 for f, c in teams)
    assert {c for _, c in teams} >= {1, 119} or len({c for _, c in teams}) > 50


def test_limit_units():
    assert rec.limit_units(1.0, 1024) == 1_000_000
    assert rec.limit_units(0.0, 1350) == 1


def test_feeder_rows_decode():
    rows = np.zeros((3, 16), dtype=np.uint64)
    rows[0, K["kFrMagic"]] = K["kFeederMagic"]
    rows[0, K["kFrStatus"]] = K["kFsTimeout"]
    rows[0, K["kFrSlot"]], rows[0, K["kFrSeq"]] = 3, 41
    rows[0, K["kFrSpanLo"]], rows[0, K["kFrSpanHi"]] = 5, 1
    rows[2, K["kFrRecSeq"]] = 2
    rows[2, K["kFrRecAcked"]] = 0xFFFFFFF0
    s = rec.feeder_summary(rec.as_int32(rows))
    assert s["magic_ok"] and s["status"] == "timeout"
    assert (s["slot"], s["seq"], s["span_cycles"]) == (3, 41, (1 << 32) + 5)
    r = rec.feeder_records(rec.as_int32(rows))
    assert r["seq"].tolist() == [0, 2] and r["acked"][1] == 0xFFFFFFF0
    assert rec.wrap(np.int64(3) - np.int64(0xFFFFFFFE)) == 5


def test_state_words_decode():
    ctl = np.zeros((2, K["KM_CTL_BYTES"] // 4), dtype=np.uint64)
    base = K["KM_CTL_STATE"] // 4
    ctl[1, base + K["kStMagic"]] = K["kStateMagic"] | 0xBEEF << 8 | 1
    word = K["kGaveStop"] | K["kWhereStall"] << 8 | 4 << 16
    ctl[1, base + K["kStReader"]], ctl[1, base + K["kStReaderSeq"]] = word, 77
    ctl[1, base + K["kStEnds"]] = K["kEndExit"] | K["kEndAbort"] << 8 | 2 << 16
    ctl[1, base + K["kStWrong"]] = 3 << 16
    st = rec.state_words(rec.as_int32(ctl))
    c = rec.core_state(st, 1, tag=0xBEEF)
    assert c["magic_ok"] and c["writer"] is None and c["compute"] is None
    assert c["reader"] == {"code": "stop", "where": "stall", "slot": 4, "seq": 77}
    assert (c["reader_end"], c["writer_end"], c["compute_end"]) == (1, 3, 2)
    assert c["compute_wrong"] == [0, 0, 3]
    assert not rec.core_state(st, 1, tag=0xBEEE)["magic_ok"]
    assert not rec.core_state(st, 0)["magic_ok"]


def test_state_words_are_distinct_and_fit():
    words = {k: v for k, v in K.items() if k.startswith("kSt") and k != "kStateWords"}
    words.pop("kStateMagic")
    taken = []
    for name, at in words.items():
        size = 16 if name in ("kStLast", "kStReported") else 1
        taken += range(at, at + size)
    assert len(taken) == len(set(taken)) == K["kStateWords"]
    assert max(taken) < K["kStateWords"]


HOST_CHECK = r"""
#include <cstdio>
#include "komorebi/device/km_record.hpp"

static km_record_t make(uint32_t seq, uint32_t seg, uint32_t first, uint32_t count) {
    km_record_t r = {};
    r.seq = r.land1 = r.land2 = r.land3 = seq;
    r.hdr = km::make_hdr(seg, 0, 0);
    r.team = km::make_team(first, count);
    return r;
}

int main() {
    km_record_t ring[KM_RING_SLOTS] = {};
    uint32_t last[KM_RING_SLOTS] = {};
    ring[3] = make(9, 1, 10, 5);   // mine (me = 12)
    ring[1] = make(8, 1, 0, 3);    // not mine
    ring[5] = make(7, 1, 12, 1);   // mine, smaller sequence
    km::Pass a = km::scan(ring, last, KM_RING_SLOTS, 12);
    ring[6] = make(10, 1, 12, 1);
    ring[6].land2 = 0;             // part-landed
    km::Pass b = km::scan(ring, last, KM_RING_SLOTS, 12);
    km::Pass c = km::scan(ring, last, 4, 12);  // depth 4: slots 0 to 3 only
    ring[0] = make(11, KM_SEG_EXIT, 0, 0);
    last[3] = 9;
    km::Pass d = km::scan(ring, last, 4, 12);
    std::printf(
        "{\"a\": [%d, %u, %d], \"b\": [%d, %u, %d], \"c\": [%d, %u, %d], "
        "\"d\": [%d, %u, %d], \"last1\": %u, \"last5\": %u, ",
        a.pick, a.pick_seq, a.partial, b.pick, b.pick_seq, b.partial, c.pick,
        c.pick_seq, c.partial, d.pick, d.pick_seq, d.partial, last[1], last[5]);
    std::printf(
        "\"newer\": [%d, %d, %d], \"in_team\": [%d, %d, %d], \"ack\": %u}\n",
        km::newer(1u, 0xFFFFFFFFu), km::newer(5u, 5u), km::newer(4u, 5u),
        km::in_team(make(1, 1, 10, 5), 14), km::in_team(make(1, 1, 10, 5), 15),
        km::in_team(make(1, 1, 10, 5), 9), KM_ACK(0x12345678u));
    km::Share s = km::share(64, 2, 60);
    std::printf("%u %u\n", s.start, s.len);
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_record_header_logic_on_the_host(tmp_path):
    src = tmp_path / "check.cpp"
    src.write_text(HOST_CHECK)
    exe = tmp_path / "check"
    cmd = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*cmd, str(src), "-o", str(exe)], check=True)
    lines = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    first, second = lines.stdout.splitlines()
    got = json.loads(first)
    assert got["a"] == [5, 7, 0]  # smallest own sequence, nothing part-landed
    assert got["last1"] == 8  # not mine: consumed
    assert got["last5"] == 0  # picked, not consumed by the scan
    assert got["b"] == [5, 7, 1]  # a part-landed record is reported
    assert got["c"] == [3, 9, 0]  # depth 4 never looks at slots 5 and 6
    assert got["d"] == [0, 11, 0]  # EXIT is taken whatever its team
    assert got["newer"] == [1, 0, 0]
    assert got["in_team"] == [1, 0, 0]
    assert got["ack"] == 0x12345678  # the whole sequence number
    assert tuple(map(int, second.split())) == rec.share(64, 2, 60) == (4, 2)


REPORT_CHECK = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/km_record.hpp"

int main(int argc, char** argv) {
    const unsigned row = static_cast<unsigned>(std::atoi(argv[1]));
    for (int k = 2; k + 1 < argc; k += 2) {
        const unsigned f = std::atoi(argv[k]), n = std::atoi(argv[k + 1]);
        km_record_t r = {};
        r.team = km::make_team(f, n);
        for (unsigned me = f; me < f + n; me++) {
            const km::Report to = km::report_to(r, me, row);
            std::printf("%u %u %u %u %u\n", f, n, me, to.parent, to.want);
        }
    }
    return 0;
}
"""

# Teams (first, count) on a grid 12 cores wide: of 1; inside one row; exactly one row;
# starting or ending mid-row; a row with a single member at either end; the whole grid.
TEAMS = [
    (0, 1),
    (23, 1),
    (118, 1),
    (0, 8),
    (3, 6),
    (0, 12),
    (12, 12),
    (11, 2),
    (6, 13),
    (5, 30),
    (11, 14),
    (24, 25),
    (0, 60),
    (7, 112),
    (0, 119),
]


def report_tree(tmp_path, row, teams):
    """{(first, count): {me: (parent, want)}} from the header, compiled on the host."""
    src = tmp_path / "report.cpp"
    src.write_text(REPORT_CHECK)
    exe = tmp_path / "report"
    cmd = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*cmd, str(src), "-o", str(exe)], check=True)
    argv = [str(exe), str(row), *(str(v) for t in teams for v in t)]
    out = subprocess.run(argv, capture_output=True, text=True, check=True)
    trees = {}
    for line in out.stdout.splitlines():
        f, n, me, parent, want = map(int, line.split())
        trees.setdefault((f, n), {})[me] = (parent, want)
    return trees


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("row", [0, 12, 10])
def test_reports_form_a_tree_rooted_at_the_first_member(tmp_path, row):
    trees = report_tree(tmp_path, row, TEAMS)
    assert sorted(trees) == sorted(TEAMS)
    for (f, n), tree in trees.items():
        assert sorted(tree) == list(range(f, f + n))
        children = {me: 0 for me in tree}
        for me, (parent, _) in tree.items():
            if me != f:
                assert parent in tree and parent < me
                children[parent] += 1
        assert all(tree[me][1] == children[me] for me in tree), (f, n)
        one_row = row == 0 or f // row == (f + n - 1) // row
        for me, (parent, _) in tree.items():
            if me == f:
                continue
            if one_row:  # one level
                assert parent == f, (f, n, me)
            else:  # a member reports in its own row; a row leader to the first member
                lead = max(f, me // row * row)
                assert parent == (lead if me != lead else f), (f, n, me)
                assert tree[parent][0] == f or parent == f  # two levels at most
        if n == 1:
            assert tree == {f: (f, 0)}


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_row_leaders_only_for_teams_that_span_rows(tmp_path):
    trees = report_tree(tmp_path, 12, TEAMS)
    assert trees[(12, 12)][12] == (12, 11)  # exactly one row: one level
    assert trees[(3, 6)][3] == (3, 5)
    assert trees[(11, 2)][12] == (11, 0)  # a row of one member: its row leader
    assert trees[(11, 2)][11] == (11, 1)
    assert trees[(11, 14)][11] == (11, 2)  # rows of 1, 12, 1: two row leaders report
    assert trees[(11, 14)][12] == (11, 11) and trees[(11, 14)][24] == (11, 0)
    assert trees[(0, 119)][0] == (0, 11 + 9)  # its own row, then nine row leaders
    assert all(trees[(0, 119)][c] == (0, 11) for c in range(12, 108, 12))
    assert trees[(0, 119)][108] == (0, 10)


OVERLAP_CHECK = r"""
#include <cstdio>
#include "komorebi/device/km_record.hpp"

int main() {
    for (unsigned fa = 0; fa < 8; fa++)
        for (unsigned na = 0; na < 6; na++)
            for (unsigned fb = 0; fb < 8; fb++)
                for (unsigned nb = 0; nb < 6; nb++)
                    std::printf("%d", km::teams_overlap(km::make_team(fa, na),
                                                        km::make_team(fb, nb)));
    std::printf(" %d\n", km::teams_overlap(KM_TEAM_ALL(119), km::make_team(118, 1)));
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_excl_overlap_is_a_shared_core(tmp_path):
    assert rec.K["KM_F_EXCL"] == 2  # flag bit 1; bit 0 stays reserved
    src = tmp_path / "overlap.cpp"
    src.write_text(OVERLAP_CHECK)
    exe = tmp_path / "overlap"
    cmd = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*cmd, str(src), "-o", str(exe)], check=True)
    got, last = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True
    ).stdout.split()
    want = "".join(
        str(int(bool(set(range(fa, fa + na)) & set(range(fb, fb + nb)))))
        for fa in range(8)
        for na in range(6)
        for fb in range(8)
        for nb in range(6)
    )
    assert got == want and last == "1"


@pytest.mark.parametrize(
    "compiler, std, suffix", [("cc", "c11", "c"), ("c++", "c++17", "cpp")]
)
def test_abi_header_compiles_as_c_and_cpp_with_its_size_checks(
    tmp_path, compiler, std, suffix
):
    if shutil.which(compiler) is None:
        pytest.skip(f"needs {compiler}")
    src = tmp_path / f"abi.{suffix}"
    src.write_text('#include "komorebi/abi/km_abi.h"\nint main(void) { return 0; }\n')
    flags = [f"-std={std}", "-Wall", "-Wextra", "-pedantic", "-Werror", "-fsyntax-only"]
    subprocess.run([compiler, *flags, "-I", str(rec.INCLUDE), str(src)], check=True)
    assert "KM_STATIC_ASSERT(sizeof(km_record_t) == 64" in rec.ABI_H.read_text()
