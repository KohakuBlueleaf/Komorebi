"""A rectangle given at each submit (the internal hook, front._sized((columns, rows))):
four command words; each core's place and each broadcast's arguments computed on the
core (km_rect.hpp), word-equal to komorebi.layout.broadcast_args on a grid of two
panels, every rectangle, both directions, both NoCs, bands; the host refusing a
rectangle past its most, by name (no card: km_rect.hpp built for the host)."""

import subprocess

import pytest
from test_class_thread_header import Pages, grid, quiet

import komorebi as km
from komorebi import layout
from komorebi.dsl.front import _sized
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

RECT = _sized((4, 4))


class Spread(km.Program):
    @km.handler
    def cast(self, x: km.Tensor, out: km.Tensor, rect: tuple = RECT):
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
    return Spread(l1=1 << 20, log=quiet, x=grid(4), out=out)


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
#include "komorebi/device/km_rect.hpp"
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
    movement = (program.built.folder / "movement.cpp").read_text()
    words = "{command.a2, command.a3, command.a4, command.a5}"
    assert f"const uint32_t rect[4] = {words};" in movement
    assert "km::rect_broadcast(km_args_row, movement.me(), km_grid, rect, true" in (
        movement
    )
    assert "kArgIncluded" not in movement
    words = program.cast.args(grid(4), grid(16, 0x9000), (2, 1, 4, 3))
    assert list(words)[2:] == [2, 1, 4, 3]
    program.rectangle["nocs"] = two_panels()
    mover_a, mover_b, common = program.args(17)
    assert mover_a == mover_b == common and len(common) == 11 and common[0] == 14


@pytest.mark.parametrize("rect", [(0, 0, 5, 1), (0, 0, 1, 5), (0, 0, 0, 1), (0, 0, 2)])
def test_a_rectangle_past_its_most_is_refused_by_name(rect):
    with pytest.raises(ProgramError, match=r"cast: rect=.*most 4 columns and 4 rows"):
        made().cast.args(grid(4), grid(16, 0x9000), rect)
