"""X280 ranks over islands (deliveries): island 0 holds them, the other islands' cores
reach them; the dispatcher word, compute's words, the run line's counts and the
refusals, without a card."""

import importlib.util
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from komorebi.islands.parts.link_plan import X280_FLAT, LinkPlanMixin
from komorebi.islands.run import IslandRun
from komorebi.program.description import Dynamic

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "src" / "komorebi_native"
EXAMPLE = ROOT / "examples" / "24_thread_rank" / "thread_rank.py"


def example_program(x280=4, hold_links=False):
    """Example 24's program (its thread on cluster 1), `x280` X280 ranks."""
    spec = importlib.util.spec_from_file_location("thread_rank", EXAMPLE)
    one = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(one)
    args = SimpleNamespace(live=1 << 13, x280=x280, deadline=10.0)
    made = one.program(args, SimpleNamespace(address=0))
    if hold_links:
        made.messages = Dynamic(**{**vars(made.messages), "hold_links": True})
    return made, one.CLUSTER


def plan(islands=2, ranks=238):
    """A link plan of `islands` islands of two cards, no region yet."""
    planned = LinkPlanMixin.__new__(LinkPlanMixin)
    planned.graph = SimpleNamespace(islands=list(range(islands)), ranks=ranks)
    planned.log = lambda line: None
    planned._regions, planned.message_links, planned.split = [], False, 1
    planned.links, planned.sessions = 1, (None,) * islands
    planned.link_cores = ()  # none: no X280 rank's send reaches another island
    return planned


def test_the_flat_rank_of_an_x280_rank_is_the_headers():
    header = (NATIVE / "include" / "komorebi" / "abi" / "km_abi.h").read_text()
    (value,) = re.findall(r"#define KM_X280_FLAT\(r\) \((0x[0-9A-F]+)u", header)
    assert int(value, 16) == X280_FLAT == 0xFFF0


def test_the_dispatcher_word_carries_the_counter_tiles_with_x280_ranks():
    with_ranks = Dynamic(handler=1, counters=2, x280_ranks=3, x280_cluster=1)
    word = with_ranks.dispatcher_word(True)
    assert (word >> 48) & 0xFF == 2  # KMD_MESSAGES_COUNTERS
    assert (word >> 32) & 0xFF == 1 | 3 << 4 and (word >> 40) & 1 == 1
    assert Dynamic(handler=1, counters=2).dispatcher_word(True) >> 41 == 0


def test_compute_counts_the_x280_ranks_another_island_holds():
    elsewhere = Dynamic(handler=1, counters=1, max_live=1 << 12)
    words = elsewhere.compute_args(10, 0, 4)
    assert words[3] == 4 and words[1] == 1 << 12  # divided by the run's ranks + 4
    held = Dynamic(handler=1, counters=1, max_live=1 << 12, x280_ranks=4)
    assert held.compute_args(10)[1:4] == words[1:4]


def test_island_1_loads_no_x280_ranks_island_0_its_ranks():
    made, cluster = example_program()
    zero, one = plan()._delivered(made, 0), plan()._delivered(made, 1)
    assert zero.messages.x280_ranks == 4 and cluster in zero.threads
    assert not getattr(zero, "x280_elsewhere", 0)
    assert one.messages.x280_ranks == 0 and one.x280_elsewhere == 4
    assert one.threads == made.threads  # the same thread: the dispatchers alike
    assert (one.messages.dispatcher_word(True) >> 32) & 0xFF == 0  # no X280 lines
    assert made.messages.x280_ranks == 4 and cluster in made.threads  # unchanged
    assert one.defines["KM_DELIVER"] == 1 and zero.defines["KM_DELIVER"] == 1
    assert one.defines["KM_X280_FAR"] == zero.defines["KM_X280_FAR"] == 4  # the rings
    assert zero.x280_far is None and not hasattr(one, "x280_far")  # no link cores


def test_island_0s_x280_ranks_get_each_island_and_its_link_core_tile():
    made, _ = example_program()
    planned = plan(islands=3, ranks=300)
    planned.graph.first = lambda island: 100 * island
    planned.graph.count = lambda island: 100
    planned.link_cores = ((7, 8), (9, 10), (11, 12))
    planned._first_link = lambda here, there: {1: 0, 2: 0}[there]  # a line: via 1
    planned.noc_of = lambda island, core: core | 2 << 8  # x | y << 8
    rows = planned._delivered(made, 0).x280_far
    assert rows == [(0, 100, 0), (100, 100, 7 | 2 << 16), (200, 100, 7 | 2 << 16)]


def test_x280_ranks_with_hold_links_are_refused_by_name():
    made, _ = example_program(hold_links=True)
    with pytest.raises(ValueError, match="X280 ranks with hold_links=True"):
        plan()._linked([made, made])


def test_x280_ranks_over_islands_of_too_many_ranks_are_refused():
    made, _ = example_program()
    with pytest.raises(ValueError, match=f"at most {X280_FLAT - 1} ranks"):
        plan(ranks=X280_FLAT)._linked([made, made])


def island_run(made):
    """An IslandRun of two islands loaded with `made` (handler numbers by name)."""
    run = IslandRun.__new__(IslandRun)
    handlers = {"grid": 1, "x280": 2}
    part = SimpleNamespace(_handler=lambda handler: (handlers[handler], 0))
    run.runs, run.program = (part, part), made
    return run


def test_a_dynamic_run_without_island_0_is_refused_by_name():
    made, _ = example_program()
    run = island_run(made)
    with pytest.raises(ValueError, match="island 0, which holds the X280 ranks"):
        run._x280_holder("grid", [(1, 119, 13)])
    run._x280_holder("grid", [(0, 112, 7), (1, 119, 7)])  # island 0's part: taken
    made.messages = Dynamic(**{**vars(made.messages), "x280_ranks": 0})
    run._x280_holder("grid", [(1, 119, 13)])  # no X280 ranks: any group


COUNTS = r"""
#include <cstdio>
#include <cstdlib>
extern "C" {
#include "x280_counts.h"
}
int main(int argc, char** argv) {
    uint32_t entries[16] = {};
    const uint32_t islands = std::strtoul(argv[1], nullptr, 0);
    for (uint32_t island = 0; island < islands; island++) {
        entries[4 * island] = std::strtoul(argv[2 + island], nullptr, 0);
    }
    const uint32_t* tail = reinterpret_cast<const uint32_t*>(entries);
    const char* const* at = argv + 2 + islands;
    const uint32_t word = kmd_x280_counts(
        tail, std::strtoul(at[0], nullptr, 0), std::strtoul(at[1], nullptr, 0),
        std::strtoul(at[2], nullptr, 0), std::strtoul(at[3], nullptr, 0),
        std::strtoul(at[4], nullptr, 0)
    );
    std::printf("%u %u\n", (word >> 8) & 0xFFu, word >> 16);
    return 0;
}
"""


@pytest.fixture(scope="module")
def counts(tmp_path_factory):
    exe = tmp_path_factory.mktemp("counts") / "counts"
    source = exe.with_suffix(".cpp")
    source.write_text(COUNTS)
    include = NATIVE / "x280" / "kernel" / "islands"
    command = ["c++", "-std=c++17", "-Wall", "-Werror", "-I", str(include)]
    subprocess.run([*command, str(source), "-o", str(exe)], check=True)

    def run(islands, group, counters, here=0):
        ranges = [first | count << 16 for first, count in islands]
        first, count = group
        low = max(i for i, (f, _) in enumerate(islands) if f <= first)
        high = max(i for i, (f, _) in enumerate(islands) if f < first + count)
        words = [len(islands), *ranges, low, high, here, first | count << 16, counters]
        out = subprocess.run(
            [str(exe), *map(str, words)], capture_output=True, text=True, check=True
        )
        return tuple(int(v) for v in out.stdout.split())

    return run


def test_the_run_line_counts_each_part_less_its_counter_tiles(counts):
    two = [(0, 119), (119, 119)]
    assert counts(two, (112, 14), 1) == (6, 12)  # 7 cores a part, a counter tile each
    assert counts(two, (112, 14), 2) == (5, 10)
    assert counts(two, (118, 14), 1) == (0, 12)  # island 0's part: its counter tile
    assert counts(two, (0, 238), 1) == (118, 236)
    assert counts(two, (0, 238), 1, here=1) == (118, 236)
    three = [(0, 119), (119, 119), (238, 119)]
    assert counts(three, (100, 200), 1) == (18, 197)  # 19, 119 and 62 cores
