"""The feeder's host helpers and the pure command logic of the device headers."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from komorebi.layout import partition

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "feeder_commands", ROOT / "tests" / "hardware" / "feeder" / "commands.py"
)
rec = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rec)
CONSTANTS = rec.CONSTANTS
GRID = (12, 10)


def test_command_fields_follow_the_abi_header():
    assert rec.COMMAND_FIELDS == [
        "sequence",
        "hdr",
        "group",
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
    assert CONSTANTS["KM_RING_SLOTS"] == 16 and CONSTANTS["KM_CONTROL_BYTES"] == 0x15C0
    assert CONSTANTS["KM_READY"] == 0xFFFFFFFF and CONSTANTS["KM_HANDLER_EXIT"] == 255
    assert CONSTANTS["KM_HANDLER_USER_MAX"] == 239 and "KM_FLAG_ACK" not in CONSTANTS
    assert (
        CONSTANTS["kStateWords"] * 4
        <= CONSTANTS["KM_CONTROL_JOB"] - CONSTANTS["KM_CONTROL_STATE"]
    )
    assert CONSTANTS["kFdArgs"] == 15 and CONSTANTS["kFeederMagic"] == 0xFEED0000


def test_start_up_arguments_are_in_header_order():
    assert rec.ARG_FIELDS[:3] == ["control", "me", "cores"]
    assert len(rec.ARG_FIELDS) == 19
    assert {"ready_base", "row", "workers"} <= set(rec.ARG_FIELDS)
    values = {name: i for i, name in enumerate(rec.ARG_FIELDS)}
    assert rec.start_args(values) == list(range(19))
    with pytest.raises(ValueError, match="missing"):
        rec.start_args({"control": 1})


def test_command_encoding():
    words = rec.command(7, 5, 60, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    fields = dict(zip(rec.COMMAND_FIELDS, words))
    assert fields["hdr"] == 7  # flags zero: every command completes
    assert fields["group"] == 5 | 60 << 16
    assert [fields[f"a{i}"] for i in range(10)] == list(range(1, 11))
    assert (
        fields["sequence"] == fields["land1"] == fields["land2"] == fields["land3"] == 0
    )
    with pytest.raises(ValueError):
        rec.command(1, 0, 1, list(range(11)))


def test_int32_round_trip_keeps_the_bits():
    rows = [[0xFFFFFFFF, 0x80000000, 5]]
    assert rec.as_uint32(rec.as_int32(rows)).tolist() == rows


@pytest.mark.parametrize("workers", [119, 120, 13, 1])
def test_groups_cover_every_worker_once(workers):
    grp, n_groups = rec.groups(GRID, workers)
    assert sorted(grp) == list(range(workers))
    leaders = {core for core, group in grp.items() if group[0] == core}
    assert n_groups == len(leaders) <= CONSTANTS["KM_RING_SLOTS"]
    for leader in leaders:
        others = [
            core for core, group in grp.items() if group[0] == leader and core != leader
        ]
        assert grp[leader][2] == len(others)
        assert sorted(grp[core][1] for core in others) == list(range(len(others)))
        assert all(
            core % GRID[0] == leader % GRID[0] and core > leader for core in others
        )
    assert (
        len({group[3] for core, group in grp.items() if group[0] == core}) == n_groups
    )


def test_partition_matches_the_layout_partition():
    for size in (0, 1, 7, 64, 120):
        for count in (1, 3, 8, 60, 119):
            for i in range(count):
                start, length = rec.partition(size, i, count)
                assert range(start, start + length) == partition(range(size), i, count)


def test_random_groups_fit_the_workers():
    rng = np.random.default_rng(3)
    groups = rec.random_groups(rng, 500, 119)
    assert all(
        count >= 1 and first >= 0 and first + count <= 119 for first, count in groups
    )
    assert {count for _, count in groups} >= {1, 119} or len(
        {count for _, count in groups}
    ) > 50


def test_limit_units():
    assert rec.limit_units(1.0, 1024) == 1_000_000
    assert rec.limit_units(0.0, 1350) == 1


def test_feeder_rows_decode():
    rows = np.zeros((3, 16), dtype=np.uint64)
    rows[0, CONSTANTS["kFrMagic"]] = CONSTANTS["kFeederMagic"]
    rows[0, CONSTANTS["kFrStatus"]] = CONSTANTS["kFsTimeout"]
    rows[0, CONSTANTS["kFrSlot"]], rows[0, CONSTANTS["kFrSequence"]] = 3, 41
    rows[0, CONSTANTS["kFrSpanLo"]], rows[0, CONSTANTS["kFrSpanHi"]] = 5, 1
    rows[2, CONSTANTS["kFrRecSequence"]] = 2
    rows[2, CONSTANTS["kFrRecAcked"]] = 0xFFFFFFF0
    summary = rec.feeder_summary(rec.as_int32(rows))
    assert summary["magic_ok"] and summary["status"] == "timeout"
    assert (summary["slot"], summary["sequence"], summary["span_cycles"]) == (
        3,
        41,
        (1 << 32) + 5,
    )
    commands = rec.feeder_commands(rec.as_int32(rows))
    assert (
        commands["sequence"].tolist() == [0, 2] and commands["acked"][1] == 0xFFFFFFF0
    )
    assert rec.wrap(np.int64(3) - np.int64(0xFFFFFFFE)) == 5


def test_state_words_decode():
    control = np.zeros((2, CONSTANTS["KM_CONTROL_BYTES"] // 4), dtype=np.uint64)
    base = CONSTANTS["KM_CONTROL_STATE"] // 4
    control[1, base + CONSTANTS["kStMagic"]] = (
        CONSTANTS["kStateMagic"] | 0xBEEF << 8 | 1
    )
    word = CONSTANTS["kGaveStop"] | CONSTANTS["kWhereStall"] << 8 | 4 << 16
    (
        control[1, base + CONSTANTS["kStReader"]],
        control[1, base + CONSTANTS["kStReaderSequence"]],
    ) = (
        word,
        77,
    )
    control[1, base + CONSTANTS["kStEnds"]] = (
        CONSTANTS["kEndExit"] | CONSTANTS["kEndAbort"] << 8 | 2 << 16
    )
    control[1, base + CONSTANTS["kStWrong"]] = 3 << 16
    state_words = rec.state_words(rec.as_int32(control))
    core = rec.core_state(state_words, 1, tag=0xBEEF)
    assert core["magic_ok"] and core["writer"] is None and core["compute"] is None
    assert core["reader"] == {
        "code": "stop",
        "where": "stall",
        "slot": 4,
        "sequence": 77,
    }
    assert (core["reader_end"], core["writer_end"], core["compute_end"]) == (1, 3, 2)
    assert core["compute_wrong"] == [0, 0, 3]
    assert not rec.core_state(state_words, 1, tag=0xBEEE)["magic_ok"]
    assert not rec.core_state(state_words, 0)["magic_ok"]


def test_state_words_are_distinct_and_fit():
    words = {
        constant: value
        for constant, value in CONSTANTS.items()
        if constant.startswith("kSt") and constant != "kStateWords"
    }
    words.pop("kStateMagic")
    taken = []
    for name, offset in words.items():
        size = 16 if name in ("kStLast", "kStReported") else 1
        taken += range(offset, offset + size)
    assert len(taken) == len(set(taken)) == CONSTANTS["kStateWords"]
    assert max(taken) < CONSTANTS["kStateWords"]


HOST_CHECK = r"""
#include <cstdio>
#include "komorebi/device/km_command.hpp"

static km_command_t make(
    uint32_t sequence, uint32_t handler, uint32_t first, uint32_t count
) {
    km_command_t r = {};
    r.sequence = r.land1 = r.land2 = r.land3 = sequence;
    r.hdr = km::make_hdr(handler, 0, 0);
    r.group = km::make_group(first, count);
    return r;
}

int main() {
    km_command_t ring[KM_RING_SLOTS] = {};
    uint32_t last[KM_RING_SLOTS] = {};
    ring[3] = make(9, 1, 10, 5);   // mine (me = 12)
    ring[1] = make(8, 1, 0, 3);    // not mine
    ring[5] = make(7, 1, 12, 1);   // mine, smaller sequence
    km::Pass a = km::scan(ring, last, KM_RING_SLOTS, 12);
    ring[6] = make(10, 1, 12, 1);
    ring[6].land2 = 0;             // part-landed
    km::Pass b = km::scan(ring, last, KM_RING_SLOTS, 12);
    km::Pass c = km::scan(ring, last, 4, 12);  // depth 4: slots 0 to 3 only
    ring[0] = make(11, KM_HANDLER_EXIT, 0, 0);
    last[3] = 9;
    km::Pass d = km::scan(ring, last, 4, 12);
    std::printf(
        "{\"a\": [%d, %u, %d], \"b\": [%d, %u, %d], \"c\": [%d, %u, %d], "
        "\"d\": [%d, %u, %d], \"last1\": %u, \"last5\": %u, ",
        a.pick, a.pick_sequence, a.partial, b.pick, b.pick_sequence, b.partial, c.pick,
        c.pick_sequence, c.partial, d.pick, d.pick_sequence, d.partial, last[1],
        last[5]);
    std::printf(
        "\"newer\": [%d, %d, %d], \"in_group\": [%d, %d, %d], \"ack\": %u}\n",
        km::newer(1u, 0xFFFFFFFFu), km::newer(5u, 5u), km::newer(4u, 5u),
        km::in_group(make(1, 1, 10, 5), 14), km::in_group(make(1, 1, 10, 5), 15),
        km::in_group(make(1, 1, 10, 5), 9), KM_ACK(0x12345678u));
    km::Partition s = km::partition(64, 2, 60);
    std::printf("%u %u\n", s.start, s.len);
    km_command_t p = make(1, 9, 0, 1);
    p.hdr = km::make_hdr(9, KM_FLAG_RESULT, 3, 200);  // the program field: bits 24-31
    std::printf(
        "%u %u %u %u %u\n", km::handler(p), km::flags(p), km::slot(p), km::program(p),
        km::program(make(1, 9, 0, 1)));
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_command_header_logic_on_the_host(tmp_path):
    source_file = tmp_path / "check.cpp"
    source_file.write_text(HOST_CHECK)
    exe = tmp_path / "check"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    lines = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    first, second, third = lines.stdout.splitlines()
    got = json.loads(first)
    assert got["a"] == [5, 7, 0]  # smallest own sequence, nothing part-landed
    assert got["last1"] == 8  # not mine: consumed
    assert got["last5"] == 0  # picked, not consumed by the scan
    assert got["b"] == [5, 7, 1]  # a part-landed command is reported
    assert got["c"] == [3, 9, 0]  # depth 4 never looks at slots 5 and 6
    assert got["d"] == [0, 11, 0]  # EXIT is taken whatever its group
    assert got["newer"] == [1, 0, 0]
    assert got["in_group"] == [1, 0, 0]
    assert got["ack"] == 0x12345678  # the whole sequence number
    assert tuple(map(int, second.split())) == rec.partition(64, 2, 60) == (4, 2)
    flag = CONSTANTS["KM_FLAG_RESULT"]
    assert tuple(map(int, third.split())) == (9, flag, 3, 200, 0)


REPORT_CHECK = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/km_command.hpp"

int main(int argc, char** argv) {
    const unsigned row = static_cast<unsigned>(std::atoi(argv[1]));
    for (int k = 2; k + 1 < argc; k += 2) {
        const unsigned f = std::atoi(argv[k]), n = std::atoi(argv[k + 1]);
        km_command_t r = {};
        r.group = km::make_group(f, n);
        for (unsigned me = f; me < f + n; me++) {
            const km::Report to = km::report_to(r, me, row);
            std::printf("%u %u %u %u %u\n", f, n, me, to.parent, to.want);
        }
    }
    return 0;
}
"""

# Groups (first, count) on a grid 12 cores wide: of 1; inside one row; exactly one row;
# starting or ending mid-row; a row with a single core at either end; the whole grid.
GROUPS = [
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


def report_tree(tmp_path, row, groups):
    """{(first, count): {me: (parent, want)}} from the header, compiled on the host."""
    source_file = tmp_path / "report.cpp"
    source_file.write_text(REPORT_CHECK)
    exe = tmp_path / "report"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    argv = [str(exe), str(row), *(str(value) for group in groups for value in group)]
    out = subprocess.run(argv, capture_output=True, text=True, check=True)
    trees = {}
    for line in out.stdout.splitlines():
        first, count, me, parent, want = map(int, line.split())
        trees.setdefault((first, count), {})[me] = (parent, want)
    return trees


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
@pytest.mark.parametrize("row", [0, 12, 10])
def test_reports_form_a_tree_rooted_at_the_first_rank(tmp_path, row):
    trees = report_tree(tmp_path, row, GROUPS)
    assert sorted(trees) == sorted(GROUPS)
    for (first, count), tree in trees.items():
        assert sorted(tree) == list(range(first, first + count))
        children = {me: 0 for me in tree}
        for me, (parent, _) in tree.items():
            if me != first:
                assert parent in tree and parent < me
                children[parent] += 1
        assert all(tree[me][1] == children[me] for me in tree), (first, count)
        one_row = row == 0 or first // row == (first + count - 1) // row
        for me, (parent, _) in tree.items():
            if me == first:
                continue
            if one_row:  # one level
                assert parent == first, (first, count, me)
            else:  # a core reports in its own row; a row leader to the first core
                lead = max(first, me // row * row)
                assert parent == (lead if me != lead else first), (first, count, me)
                assert tree[parent][0] == first or parent == first  # two levels at most
        if count == 1:
            assert tree == {first: (first, 0)}


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_row_leaders_only_for_groups_that_span_rows(tmp_path):
    trees = report_tree(tmp_path, 12, GROUPS)
    assert trees[(12, 12)][12] == (12, 11)  # exactly one row: one level
    assert trees[(3, 6)][3] == (3, 5)
    assert trees[(11, 2)][12] == (11, 0)  # a row of one core: its row leader
    assert trees[(11, 2)][11] == (11, 1)
    assert trees[(11, 14)][11] == (11, 2)  # rows of 1, 12, 1: two row leaders report
    assert trees[(11, 14)][12] == (11, 11) and trees[(11, 14)][24] == (11, 0)
    assert trees[(0, 119)][0] == (0, 11 + 9)  # its own row, then nine row leaders
    assert all(trees[(0, 119)][core] == (0, 11) for core in range(12, 108, 12))
    assert trees[(0, 119)][108] == (0, 10)


OVERLAP_CHECK = r"""
#include <cstdio>
#include "komorebi/device/km_command.hpp"

int main() {
    for (unsigned fa = 0; fa < 8; fa++)
        for (unsigned na = 0; na < 6; na++)
            for (unsigned fb = 0; fb < 8; fb++)
                for (unsigned nb = 0; nb < 6; nb++)
                    std::printf("%d", km::groups_overlap(km::make_group(fa, na),
                                                        km::make_group(fb, nb)));
    std::printf(" %d\n", km::groups_overlap(KM_GROUP_ALL(119), km::make_group(118, 1)));
    return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_excl_overlap_is_a_shared_core(tmp_path):
    assert rec.CONSTANTS["KM_FLAG_EXCLUSIVE"] == 2  # flag bit 1; bit 0 stays reserved
    source_file = tmp_path / "overlap.cpp"
    source_file.write_text(OVERLAP_CHECK)
    exe = tmp_path / "overlap"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(rec.INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    got, last = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True
    ).stdout.split()
    want = "".join(
        str(
            int(
                bool(
                    set(range(first_a, first_a + count_a))
                    & set(range(first_b, first_b + count_b))
                )
            )
        )
        for first_a in range(8)
        for count_a in range(6)
        for first_b in range(8)
        for count_b in range(6)
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
    source_file = tmp_path / f"abi.{suffix}"
    source_file.write_text(
        '#include "komorebi/abi/km_abi.h"\nint main(void) { return 0; }\n'
    )
    flags = [f"-std={std}", "-Wall", "-Wextra", "-pedantic", "-Werror", "-fsyntax-only"]
    subprocess.run(
        [compiler, *flags, "-I", str(rec.INCLUDE), str(source_file)], check=True
    )
    assert "KM_STATIC_ASSERT(sizeof(km_command_t) == 64" in rec.ABI_H.read_text()
