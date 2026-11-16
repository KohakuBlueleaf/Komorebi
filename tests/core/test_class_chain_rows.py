"""A chain call whose roots write several tile rows: its record in tile columns, its
readers' rows proven by the host; refused, each by its form: a row not written, roots of
different rows, pages, km.past outside the rows, km.direct (in place at the make), a
row loop whose rectangles make no rectangle (no card)."""

import pytest
from chain_fixture import CORES, Pages, quiet, run_of

import komorebi as km
from komorebi import chain as kc
from komorebi.errors import ProgramError
from komorebi.tensor import Tensor

R = 2  # the tile rows a root writes
G = 4  # a shared reader's group


def grid(rows, columns, address):
    """A float32 tensor of rows x columns tiles, a page a tile."""
    return Tensor(
        Pages(4096, rows * columns, address),
        (32 * rows, 32 * columns),
        "float32",
        "tile",
    )


class Rows(km.Program):
    """own: core g copies rows 0..R-1 of column g (looped: a row a turn; gapped: rows
    0, 2; staggered: from row g % 2); twos, top, deep: sums of t's columns 2g, 2g + 1
    over rows 0..R-1, row 0, rows 0..R; paged: page g; share_sum: a group's share up
    its tree; step, back: t's column g, back after u and km.past(t)."""

    @km.handler
    def own(self, s: km.Tensor, t: km.Tensor):
        out = km.Stream(t.tiles[0:R, self.rank : self.rank + 1], write=True)
        for tile in km.Stream(s.tiles[0:R, self.rank : self.rank + 1]):
            out.push(tile)

    @km.handler
    def looped(self, s: km.Tensor, t: km.Tensor):
        for r in range(R):
            total = km.tile.zeros(km.fp32)
            for tile in km.Stream(s.tiles[r : r + 1, self.rank : self.rank + 1]):
                total += tile
            t.tiles[r : r + 1, self.rank : self.rank + 1].write(total)

    @km.handler
    def gapped(self, s: km.Tensor, t: km.Tensor):
        for r in range(R):
            total = km.tile.zeros(km.fp32)
            for tile in km.Stream(s.tiles[r : r + 1, self.rank : self.rank + 1]):
                total += tile
            t.tiles[2 * r : 2 * r + 1, self.rank : self.rank + 1].write(total)

    @km.handler
    def staggered(self, s: km.Tensor, t: km.Tensor):
        top = self.rank % 2
        out = km.Stream(t.tiles[top : top + R, self.rank : self.rank + 1], write=True)
        for tile in km.Stream(s.tiles[top : top + R, self.rank : self.rank + 1]):
            out.push(tile)

    @km.handler
    def twos(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.tiles[0:R, 2 * self.rank : 2 * self.rank + 2], block=2):
            total += tile
        y.pages(self.rank, 1).write(total)

    @km.handler
    def top(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.tiles[0:1, 2 * self.rank : 2 * self.rank + 2], block=2):
            total += tile
        y.pages(self.rank, 1).write(total)

    @km.handler
    def deep(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(
            t.tiles[0 : R + 1, 2 * self.rank : 2 * self.rank + 2], block=2
        ):
            total += tile
        y.pages(self.rank, 1).write(total)

    @km.handler
    def paged(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.pages(self.rank, 1)):
            total += tile
        y.pages(self.rank, 1).write(total)

    @km.handler
    def share_sum(self, t: km.Tensor, y: km.Tensor):
        share = self.rank % G
        k0 = share * 2
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.tiles[0:R, k0 : k0 + 2], block=2):
            total += tile
        whole = km.reduce(total, form="tree", group=G)
        if self.rank == self.leader:
            y.pages(self.rank // G, 1).write(whole)

    @km.handler
    def step(self, t: km.Tensor, u: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.tiles[0:R, self.rank : self.rank + 1]):
            total += tile
        u.pages(self.rank, 1).write(total)

    @km.handler
    def back(self, u: km.Tensor, t: km.Tensor, y: km.Tensor, rows: int):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(u.pages(self.rank, 1)):
            total += tile
        for tile in km.Stream(t.tiles[0:rows, self.rank : self.rank + 1]):
            total += tile
        y.pages(self.rank, 1).write(total)


def made():
    t = {
        "s": grid(R + 1, CORES, 0x10000),
        "t": grid(R + 1, CORES, 0x20000),
        "u": grid(1, CORES, 0x30000),
        "y": grid(1, CORES, 0x40000),
    }
    return Rows(l1=1 << 20, log=quiet, chain=True, **t), t


def chain(calls):
    program, _ = made()
    return run_of(program).chain(calls, cores=(0, CORES))


def test_a_root_of_several_rows_is_a_block_of_columns_and_its_readers_ranges_too():
    program, t = made()
    calls = [program.own(t["s"], t["t"]), program.twos(t["t"], t["y"])]
    own, twos = run_of(program).chain(calls, cores=(0, CORES)).calls
    assert own.width == 1
    assert twos.ranges == (kc.Range(0, 2, 2),)


def test_a_reader_of_fewer_rows_than_written_is_described_the_same():
    program, t = made()
    calls = [program.own(t["s"], t["t"]), program.top(t["t"], t["y"])]
    assert run_of(program).chain(calls, cores=(0, CORES)).calls[1].ranges == (
        kc.Range(0, 2, 2),
    )


def test_a_shared_reader_takes_its_share_of_the_columns():
    program, t = made()
    calls = [program.own(t["s"], t["t"]), program.share_sum(t["t"], t["y"])]
    shared = run_of(program).chain(calls, cores=(0, CORES)).calls[1]
    assert shared.size == G and shared.reads == ()
    assert shared.k == (0, 2 * G)


def test_a_read_of_a_row_the_roots_do_not_write_is_refused():
    program, t = made()
    with pytest.raises(kc.ChainRefused, match=r"row 2 of .*rows 0 to 1"):
        chain([program.own(t["s"], t["t"]), program.deep(t["t"], t["y"])])


def test_roots_that_write_different_rows_are_refused():
    program, t = made()
    with pytest.raises(kc.ChainRefused, match=r"rows 1 to 2.*rows 0 to 1"):
        chain([program.staggered(t["s"], t["t"]), program.twos(t["t"], t["y"])])


def test_pages_of_a_tensor_written_in_rows_are_refused():
    program, t = made()
    with pytest.raises(kc.ChainRefused, match=r"written in rows"):
        chain([program.own(t["s"], t["t"]), program.paged(t["t"], t["y"])])


def test_a_past_read_inside_the_rows_is_proven_and_outside_them_refused():
    program, t = made()
    calls = [
        program.own(t["s"], t["t"]),
        program.step(t["t"], t["u"]),
        program.back(t["u"], km.past(t["t"]), t["y"], R),
    ]
    assert run_of(program).chain(calls, cores=(0, CORES)).calls[2].reads[0].earlier
    calls[2] = program.back(t["u"], km.past(t["t"]), t["y"], R + 1)
    with pytest.raises(kc.ChainRefused, match=r"row 2 of .*rows 0 to 1"):
        chain(calls)


def test_a_row_loops_rectangles_are_one_and_a_gap_between_them_is_refused():
    program, t = made()
    calls = [program.looped(t["s"], t["t"]), program.twos(t["t"], t["y"])]
    looped, twos = run_of(program).chain(calls, cores=(0, CORES)).calls
    assert looped.width == 1 and twos.ranges == (kc.Range(0, 2, 2),)
    with pytest.raises(kc.ChainRefused, match=r"loop's r make no rectangle"):
        chain([program.gapped(t["s"], t["t"]), program.twos(t["t"], t["y"])])


class Direct(Rows):
    """twos taking t direct (a copy); streamed: own by a write stream."""

    twos = km.handler(direct=("t",))(Rows.twos.function)

    @km.handler
    def streamed(self, s: km.Tensor, t: km.Tensor):
        out = km.Stream(t.tiles[0:R, self.rank : self.rank + 1], write=True)
        for tile in km.Stream(s.tiles[0:R, self.rank : self.rank + 1]):
            out.push(tile)


class Halves(Rows):
    """twos taking t direct in its stream's halves."""

    twos = km.handler(direct=("t",), in_place=True)(Rows.twos.function)


@pytest.mark.parametrize(
    ("cls", "writer", "why"),
    [
        (Direct, "looped", "not built: drop km.direct"),
        (Direct, "streamed", r"write stream \(line \d+\).*drop km.direct"),
    ],
)
def test_a_direct_input_written_in_rows_is_refused_by_its_form(cls, writer, why):
    _, t = made()
    program = cls(l1=1 << 20, log=quiet, chain=True, **t)
    calls = [
        getattr(program, writer)(t["s"], t["t"]),
        program.twos(km.direct(t["t"]), t["y"]),
    ]
    with pytest.raises(kc.ChainRefused, match=why):
        run_of(program).chain(calls, cores=(0, CORES))


def test_an_in_place_input_over_rows_is_refused_at_the_make():
    _, t = made()
    with pytest.raises(ProgramError, match="in place takes one stream of its pages"):
        Halves(l1=1 << 20, log=quiet, chain=True, **t)
