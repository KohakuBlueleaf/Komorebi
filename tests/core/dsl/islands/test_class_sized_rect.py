"""A rectangle given at each submit, km.UPTO(rect=(columns, rows)): four command words;
km_rect.hpp built for the host, word-equal to layout.broadcast_args on a grid of two
panels (rows, columns, both NoCs, bands); the host's refusals by name (no card)."""

import subprocess
from types import SimpleNamespace

import pytest
from class_thread_header_fixture import Pages, grid, quiet

import komorebi as km
from komorebi import layout
from komorebi.device.tensor import Tensor
from komorebi.dsl.program import rect_grid_args
from komorebi.errors import ProgramError
from komorebi.session.run import Run


class Spread(km.Program):
    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor):
        row = km.Broadcast(self, along="row", window=2)
        if row.sender():  # noqa: SIM108 (the compiler's form)
            tiles = row.cast(x.pages(self.y, 1).read())
        else:
            tiles = row.arrive()
        total = km.tile.zeros(km.fp32)
        for tile in tiles:
            total += tile
        out.pages(self.y * 4 + self.x, 1).write(total)


def made():
    out = Tensor(Pages(2048, 16, 0x9000), (32, 16 * 32), "bfloat16", "tile")
    up_to = {"cast": [km.UPTO(rect=(4, 4))]}
    return Spread(l1=1 << 20, log=quiet, x=grid(4), out=out, handlers=up_to)


def two_panels():
    """A grid of 14 columns of two panels (NoC0 x 1 to 7, 10 to 16) and 10 rows,
    its translated words apart from its NoC0 words."""
    columns, rows = 14, 10
    xs = [*range(1, 8), *range(10, 17)]
    order = layout.core_order((columns, rows))
    noc0 = [xs[x] | (y + 2) << 16 for x, y in order]
    table = [(xs[x] + 0x10) | (y + 0x12) << 16 for x, y in order]
    return layout.nocs((columns, rows), table, noc0)


DRIVER = r"""
#include <cstdio>
#include "komorebi/device/movement/km_rect.hpp"
struct Args { uint32_t rank, receivers, sender, mc_start, mc_end, noc; };
int main() {
    uint32_t words[11];
    for (auto& w : words) { if (scanf("%u", &w) != 1) return 1; }
    const km::RectGrid grid = km::rect_grid(words);
    uint32_t rect[4], row, noc, length, core;
    while (scanf("%u %u %u %u %u %u %u %u", rect, rect + 1, rect + 2, rect + 3, &row,
                 &noc, &length, &core) == 8) {
        km::RectPlace p = km::rect_place(core, grid.width, rect);
        Args a{};
        a.noc = noc;
        km::rect_broadcast(a, core, grid, rect, row != 0, length, 0xFFu);
        printf("%u %u %u %u %u %u %u %u %u\n", p.x, p.y, p.included, a.rank,
               a.receivers, a.sender, a.mc_start, a.mc_end, 0u);
    }
    return 0;
}
"""


def test_the_core_computes_the_host_s_arguments():
    nocs = two_panels()
    columns, rows = nocs.grid
    program = made()
    program.rectangle["nocs"] = nocs
    words = program.args(0)[2]
    folder = program.built.folder  # the build's own folder
    source = folder / "driver.cpp"
    source.write_text(DRIVER)
    include = layout.__file__.rsplit("/", 1)[0] + "/../komorebi_native/include"
    built = folder / "driver"
    subprocess.run(
        ["g++", "-std=c++17", "-O1", "-I", include, str(source), "-o", str(built)],
        check=True,
    )
    cases, want = [], []
    fields = ("rank", "receivers", "sender", "mc_start", "mc_end")
    rects = [(0, 0, columns, rows), (5, 2, 4, 3), (6, 0, 2, 10), (3, 7, 1, 1)]
    rects += [(1, 1, 9, 2), (8, 4, 6, 6)]
    for rect in rects:
        x0, y0, w, h = rect
        first = y0 * columns + x0
        group = layout.Group(first, (y0 + h - 1) * columns + x0 + w - first)
        for along in ("row", "column"):
            for noc in (0, 1):
                for length in (0, 2, 3):
                    host = layout.broadcast_args(
                        nocs.grid, nocs.panels, group, rect, along, noc, nocs.noc0,
                        nocs.noc_xy, (0, 0, 0, 0), length or None,
                    )  # fmt: skip
                    for core in range(columns * rows):
                        cx, cy = core % columns, core // columns
                        inside = x0 <= cx < x0 + w and y0 <= cy < y0 + h
                        row = int(along == "row")
                        cases.append(f"{x0} {y0} {w} {h} {row} {noc} {length} {core}")
                        if inside:
                            args = host[core]
                            got = [cx - x0, cy - y0, 1, *(args[n] for n in fields)]
                        else:
                            got = [0, 0, 0, 0xFF, 0, 0, 0, 0]
                        want.append([*got, 0])
    out = subprocess.run(
        [str(built)],
        input=" ".join(map(str, words)) + "\n" + "\n".join(cases) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    wrong = [
        (case, line, w)
        for case, line, w in zip(cases, out, want, strict=False)
        if [int(n) for n in line.split()] != w
    ]
    assert len(out) - 1 == len(cases)
    assert not wrong, (len(wrong), wrong[:3])


def test_the_rectangle_is_four_words_and_the_core_s_place_its_own():
    program = made()
    words = program.cast.args(grid(4), grid(16, 0x9000), (2, 1, 4, 3))
    assert list(words)[2:] == [2, 1, 4, 3]
    program.rectangle["nocs"] = two_panels()
    mover_a, mover_b, common = program.args(17)
    assert mover_a == mover_b == common and len(common) == 11 and common[0] == 14


@pytest.mark.parametrize("rect", [(0, 0, 5, 1), (0, 0, 1, 5), (0, 0, 0, 1), (0, 0, 2)])
def test_a_rectangle_past_its_most_is_refused_by_name(rect):
    with pytest.raises(ProgramError, match=r"cast: rect=.*most 4 columns and 4 rows"):
        made().cast.args(grid(4), grid(16, 0x9000), rect)


def test_a_rectangle_given_at_each_submit_takes_no_rect_at_the_make():
    up_to = {"cast": [km.UPTO(rect=(4, 4))]}
    with pytest.raises(ProgramError, match=r"the make takes no rect="):
        Spread(l1=1 << 20, log=quiet, x=grid(4), out=grid(16, 0x9000),
               handlers=up_to, rect=(0, 0, 4, 4))  # fmt: skip


# Islands of the 14 x 10 grid of two_panels: two cards (one below the other, the
# link cores last: 138 ranks each) and a card split in 3 (rows 0-3, 4-6, 7-9).
CARDS = {"bands": None, "tops": [0, 10], "rows": [10, 10], "ranks": [138, 138]}
SPLIT = {"bands": [(0, 4), (4, 3), (7, 3)], "tops": [0, 4, 7], "rows": [4, 3, 3]}
SPLIT["ranks"] = [55, 40, 41]


def stack_of(kind, nocs):
    firsts = [sum(kind["ranks"][:i]) for i in range(len(kind["ranks"]))]
    islands = [(nocs, first, n) for first, n in zip(firsts, kind["ranks"])]
    return {k: kind[k] for k in ("bands", "tops", "rows")} | {"islands": islands}


ISLANDS_DRIVER = r"""
#include <cstdio>
#include "komorebi/device/movement/km_rect.hpp"
struct Args {
    uint32_t rank, receivers, sender, mc_start, mc_end, noc, next, fed, from;
};
int main() {
    uint32_t count, words[64];
    if (scanf("%u", &count) != 1) return 1;
    for (uint32_t i = 0; i < 11 + count; i++) {
        if (scanf("%u", words + i) != 1) return 1;
    }
    const km::RectGrid grid = km::rect_grid(words);
    const km::RectIslands islands = km::rect_islands(words + 11, count);
    uint32_t rect[4], row, noc, length, rank;
    while (scanf("%u %u %u %u %u %u %u %u", rect, rect + 1, rect + 2, rect + 3, &row,
                 &noc, &length, &rank) == 8) {
        const uint32_t whole = km::rect_whole(rank, grid.width, islands);
        km::RectPlace p = km::rect_place(whole, grid.width, rect);
        Args a{};
        a.noc = noc;
        const bool row_ = row != 0;
        km::rect_broadcast_islands(a, rank, grid, islands, rect, row_, length, 0xFFu);
        printf("%u %u %u %u %u %u %u %u %u %u %u\n", p.x, p.y, p.included, a.rank,
               a.receivers, a.sender, a.mc_start, a.mc_end, a.next, a.fed, a.from);
    }
    return 0;
}
"""


@pytest.mark.parametrize("kind", ["cards", "split"])
def test_on_islands_the_core_computes_the_host_s_parted_arguments(kind, tmp_path):
    """km_rect.hpp built for the host, word-equal to layout.island_broadcast_args (the
    form examples/03 writes by hand) for rectangles that run on across islands: each
    rank's place in the whole, its part's rank, sender (a core of its card) and
    corners, the next part's first flat rank, the feeder's cores and first rank."""
    nocs = two_panels()
    columns = nocs.grid[0]
    kind = CARDS if kind == "cards" else SPLIT
    stack = stack_of(kind, nocs)
    count = len(kind["ranks"])
    compiled = SimpleNamespace(name="Spread", islands=(count, 0), rect_sized=True)
    words = rect_grid_args(compiled, nocs, stack)(0)[0]
    built = tmp_path / "driver"
    source = built.with_suffix(".cpp")
    source.write_text(ISLANDS_DRIVER)
    include = layout.__file__.rsplit("/", 1)[0] + "/../komorebi_native/include"
    subprocess.run(
        ["g++", "-std=c++17", "-O1", "-I", include, str(source), "-o", str(built)],
        check=True,
    )
    if kind is CARDS:
        rects = [
            (0, 7, 5, 6),
            (3, 0, 9, 20),
            (6, 9, 2, 2),
            (8, 2, 4, 15),
            (0, 0, 12, 10),
        ]
    else:
        rects = [(0, 0, 12, 10), (2, 3, 4, 5), (5, 6, 7, 1), (9, 1, 3, 9), (6, 2, 1, 2)]
    cases, want = [], []
    fields = ("rank", "receivers", "sender", "mc_start", "mc_end", "next", "fed")
    for rect in rects:
        x0, y0, w, h = rect
        first = layout.stack_rank(stack, columns, x0, y0)
        last = layout.stack_rank(stack, columns, x0 + w - 1, y0 + h - 1)
        group = layout.Group(first, last + 1 - first)
        for along in ("row", "column"):
            for noc in (0, 1):
                for length in (0, 2, 3):
                    host = layout.island_broadcast_args(
                        stack["islands"], group, rect, along, noc, (0, 0, 0, 0),
                        "below", stack["bands"], length or None,
                    )  # fmt: skip
                    for rank in group.indices:
                        island = max(
                            i
                            for i, one in enumerate(stack["islands"])
                            if one[1] <= rank
                        )
                        local = rank - stack["islands"][island][1]
                        cx = local % columns
                        cy = stack["tops"][island] + local // columns
                        inside = x0 <= cx < x0 + w and y0 <= cy < y0 + h
                        row = int(along == "row")
                        cases.append(f"{x0} {y0} {w} {h} {row} {noc} {length} {rank}")
                        args = host[island][local]
                        place = [cx - x0, cy - y0, 1] if inside else [0, 0, 0]
                        got = [*place, *(args[n] for n in fields), args["from"]]
                        want.append(got)
    out = subprocess.run(
        [str(built)],
        input=f"{count} " + " ".join(map(str, words)) + "\n" + "\n".join(cases) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\n")
    wrong = [
        (case, line, w)
        for case, line, w in zip(cases, out, want, strict=False)
        if [int(n) for n in line.split()] != w
    ]
    assert len(out) - 1 == len(cases)
    assert not wrong, (len(wrong), wrong[:3])
    parted = [w for w in want if w[8] != layout.BROADCAST_NO_NEXT or w[9]]
    assert parted  # parts fed across were among the cases


def test_on_islands_a_rectangle_given_at_a_submit_counts_from_its_first_island():
    """The rectangle counts from the island of the group's first rank and runs on to
    the next (the stack's rows, given to the cores so): on a split card the card's
    rows; over cards island 1's rows from row 10. Every cell a rank, its group the
    rectangle's first to last flat rank."""
    program = made()
    method = program.cast.method
    words = list(program.cast.args(grid(4), grid(16, 0x9000), (2, 1, 4, 2)))
    at = method.words.index(("rect", "rect"))
    nocs = two_panels()
    split = SimpleNamespace(
        grid=(14, 10),
        programs=[SimpleNamespace(rectangle={"stack": stack_of(SPLIT, nocs)})],
    )
    given = Run._check_given_rectangle(split, method, words, (55 + 16, 18))
    assert given[at : at + 4] == [2, 5, 4, 2]  # island 1 from the card's row 4
    words[at + 1] = 2  # island 1's rows 2 and 3: its row 6, island 2's row 7
    given = Run._check_given_rectangle(split, method, words, (55 + 30, 16))
    assert given[at : at + 4] == [2, 6, 4, 2]
    cards = SimpleNamespace(
        grid=(14, 10),
        programs=[SimpleNamespace(rectangle={"stack": stack_of(CARDS, nocs)})],
    )
    words[at + 1] = 9  # island 0's row 9, island 1's row 0
    given = Run._check_given_rectangle(cards, method, words, (128, 16))
    assert given[at : at + 4] == [2, 9, 4, 2]
    with pytest.raises(
        ValueError, match=r"rect=\(2, 9, 4, 2\) is the group \(128, 16\)"
    ):
        Run._check_given_rectangle(cards, method, words, (128, 17))
    with pytest.raises(ValueError, match=r"from island 1's first row is off the "):
        Run._check_given_rectangle(cards, method, words, (138 + 128, 16))
    words[at + 1] = 1
    given = Run._check_given_rectangle(cards, method, words, (138 + 16, 18))
    assert given[at : at + 4] == [2, 11, 4, 2]  # island 1's row 1: the stack's 11
    words[at : at + 4] = [10, 9, 4, 2]  # island 0's link cores, columns 12 and 13
    with pytest.raises(ValueError, match=r"takes a link core of an island"):
        Run._check_given_rectangle(cards, method, words, (136, 18))
    words[at : at + 4] = [2, 1, 4, 8]  # the split's rows 1 to 8: its three islands
    given = Run._check_given_rectangle(split, method, words, (16, 99))  # 16..114
    assert given[at : at + 4] == [2, 1, 4, 8]
