"""Groups and the reduce schedules of the device headers (km_group.hpp), compiled on
the host and checked against a plain Python model of the two forms."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "komorebi_native" / "include"
pytestmark = pytest.mark.skipif(shutil.which("c++") is None, reason="needs c++")

SCHEDULE = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/core/km_group.hpp"

int main(int argc, char** argv) {
    km_command_t r = {};
    r.group = km::make_group(5, std::atoi(argv[1]));
    const unsigned form = std::atoi(argv[2]);
    for (unsigned me = 5; me < 5 + km::count(r); me++) {
        const km::Group g = km::group_of(r, me);
        const km::ReduceStep s = km::reduce_send(g, form);
        std::printf("%u %d %u %u %u", g.rank, s.root, s.to, s.port, s.slot);
        for (unsigned p = 0; p < KM_PORTS; p++) {
            std::printf(" %u", km::reduce_wants(g, form, p));
        }
        std::printf("\n");
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def schedule(tmp_path_factory):
    exe = tmp_path_factory.mktemp("schedule") / "schedule"
    source_file = exe.with_suffix(".cpp")
    source_file.write_text(SCHEDULE)
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)

    def run(size, form):
        out = subprocess.run(
            [str(exe), str(size), str(form)], capture_output=True, text=True, check=True
        )
        rows = [list(map(int, line.split())) for line in out.stdout.splitlines()]
        return {
            row[0]: (bool(row[1]), row[2] - 5, row[3], row[4], row[5:]) for row in rows
        }

    return run


@pytest.mark.parametrize("size", [1, 6, 119])
def test_tree_brings_every_value_to_rank_0_once(schedule, size):
    plan = schedule(size, 1)
    held = {rank: [rank] for rank in plan}
    for level in range(7):
        sends = {
            node: entry
            for node, entry in plan.items()
            if not entry[0] and entry[2] == level
        }
        for rank, (_, receiver, port, slot, _) in sends.items():
            assert slot == port == level and rank - receiver == 1 << level
            assert plan[receiver][4][level] == 1  # the receiver waits for one on port r
            held[receiver] += held.pop(rank)
        receivers = {
            node for node, entry in plan.items() if node in held and entry[4][level]
        }
        assert receivers == {entry[1] for entry in sends.values()}
    assert list(held) == [0] and sorted(held[0]) == list(range(size))


@pytest.mark.parametrize("size", [1, 6, 119])
def test_owner_sends_every_value_to_rank_0_in_its_own_slot(schedule, size):
    plan = schedule(size, 0)
    assert plan[0][0] and plan[0][4] == [size - 1 if size > 1 else 0] + [0] * 7
    for rank in range(1, size):
        root, receiver, port, slot, wants = plan[rank]
        assert (
            not root and (receiver, port, slot) == (0, 0, rank - 1) and not any(wants)
        )


DIGITS = r"""
#include <cstdio>
#include "komorebi/device/core/km_group.hpp"

int main() {
    km_command_t r = {};
    r.group = km::make_group(10, 11);  // index = block * 4 + part; the last block short
    for (unsigned me = 10; me < 21; me++) {
        const km::Group g = km::digit_group(r, me, 1, 4);
        std::printf("%u %u %u %u\n", g.base, g.size, g.rank, km::core_of(g, g.rank));
    }
    return 0;
}
"""


def test_digit_group_is_the_ranks_of_one_block(tmp_path):
    source_file, exe = tmp_path / "digits.cpp", tmp_path / "digits"
    source_file.write_text(DIGITS)
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    rows = [tuple(map(int, line.split())) for line in out.stdout.splitlines()]
    for core, (base, size, rank, me) in enumerate(rows):
        assert base == 10 + core // 4 * 4 and rank == core % 4 and me == 10 + core
        assert size == (4 if core < 8 else 3)


PAGES = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/core/km_group.hpp"

int main(int argc, char** argv) {
    km_command_t r = {};
    r.group = km::make_group(0, std::atoi(argv[1]));
    const unsigned form = std::atoi(argv[2]);
    const km::Arrival arr = {1, 118, 2};
    const unsigned need = km::reduce_slots(km::group_of(r, 0), form);
    std::printf("%u %u\n", km::arrival_slots(arr), need);
    for (unsigned me = 1; me < km::count(r); me++) {
        const km::ReduceStep s = km::reduce_send(km::group_of(r, me), form);
        std::printf("%u %u\n", s.to, km::slot_page(arr, s.slot));
    }
    return 0;
}
"""


def test_parts_of_two_pages_land_on_their_own_pages(tmp_path):
    source_file, exe = tmp_path / "pages.cpp", tmp_path / "pages"
    source_file.write_text(PAGES)
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)

    def run(size, form):
        args = [str(exe), str(size), str(form)]
        out = subprocess.run(args, capture_output=True, text=True, check=True)
        rows = [tuple(map(int, line.split())) for line in out.stdout.splitlines()]
        return rows[0], rows[1:]

    for size, form in [(60, 0), (119, 1)]:
        (slots, needed), sends = run(size, form)
        assert slots == 59 and needed <= slots
        pages = {}  # receiver -> pages its senders fill, 2 per part
        for destination, first in sends:
            got = pages.setdefault(destination, set())
            assert first % 2 == 0 and first + 2 <= 118 and not {first, first + 1} & got
            got |= {first, first + 1}
    assert run(61, 0)[0] == (59, 60)  # by owner, 61 cores need 60 slots: too many


SENDERS = r"""
#include <cstdio>
#include <cstdlib>
#include "komorebi/device/core/km_group.hpp"

int main(int argc, char** argv) {
    km_command_t r = {};
    r.group = km::make_group(5, std::atoi(argv[1]));
    const unsigned form = std::atoi(argv[2]);
    for (unsigned me = 5; me < 5 + km::count(r); me++) {
        const km::Group g = km::group_of(r, me);
        for (unsigned p = 0; p < KM_PORTS; p++) {
            const km::ReduceSenders s = km::reduce_senders_flat(g, form, p);
            for (unsigned rank = s.first; rank < s.end; rank++) {
                std::printf("%u %u %u\n", g.rank, p, rank);
            }
        }
    }
    return 0;
}
"""


@pytest.mark.parametrize("form", [0, 1])
@pytest.mark.parametrize("size", [1, 2, 6, 119, 120])
def test_a_repeated_reduce_credits_exactly_the_ranks_that_send_to_it(
    schedule, tmp_path, form, size
):
    """The credit of a reduce's turn 2 on goes to each rank whose send lands on this
    rank's port: the inverse of reduce_send, every sender once."""
    source_file, exe = tmp_path / "senders.cpp", tmp_path / "senders"
    source_file.write_text(SENDERS)
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(INCLUDE)]
    subprocess.run([*command, str(source_file), "-o", str(exe)], check=True)
    args = [str(exe), str(size), str(form)]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    credited = [tuple(map(int, line.split())) for line in out.stdout.splitlines()]
    plan = schedule(size, form)
    sends = {
        (receiver, port, rank)
        for rank, (root, receiver, port, _, _) in plan.items()
        if not root
    }
    assert sorted(credited) == sorted(sends) and len(set(credited)) == len(credited)
