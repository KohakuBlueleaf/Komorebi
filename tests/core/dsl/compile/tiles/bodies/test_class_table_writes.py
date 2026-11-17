"""A write at a table's words in a chain (zbuf.pages(e.first, e.pages) in an entries
loop): anywhere in its tensor, the next call waits for every block; refused: a read
of it further on, and a call that writes the tensor by pages too (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, quiet, run_of, tensor

import komorebi as km
from komorebi.chain import calls as kc


class Span(km.Entry):
    first: int
    pages: int


@km.tile_function
def coverage(c):
    return km.minimum(km.maximum(c, 0.0), 1.0)


class Raster(km.Program):
    @km.handler
    def raster(self, counts: km.Tensor, lists: km.Tensor, out: km.Tensor):
        for e in counts.pages(self.rank, 1).entries(Span, first=0, size=2, fixed=1):
            results = km.Stream(out.pages(e.first, e.pages), write=True)
            for tile in km.Stream(lists.pages(e.first, e.pages)):
                results.push(coverage(tile))

    @km.handler
    def twice(self, counts: km.Tensor, lists: km.Tensor, out: km.Tensor):
        for e in counts.pages(self.rank, 1).entries(Span, first=0, size=2, fixed=1):
            results = km.Stream(out.pages(e.first, e.pages), write=True)
            for tile in km.Stream(lists.pages(e.first, e.pages)):
                results.push(coverage(tile))
        mine = km.Stream(out.pages(self.rank, 1), write=True)
        for tile in km.Stream(lists.pages(self.rank, 1)):
            mine.push(tile)

    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)


def given():
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate("slcoz")}


def made(t):
    return Raster(l1=1 << 20, log=quiet, counts=t["c"], lists=t["l"], out=t["o"],
                  a=t["o"], chain=True)  # fmt: skip


def test_the_next_call_waits_for_every_block_of_a_write_at_a_table_s_words():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [program.raster(t["c"], t["l"], t["o"]), program.copy(t["o"], t["z"], WIDTH)],
        cores=(0, CORES),
    )
    raster, copy = chain.calls
    assert (raster.output, raster.width) == ("out (0x40000)", 1)  # a tile a core
    assert copy.input == raster.output
    assert copy.ranges == (kc.Range(0, 0, CORES),)  # every core's mark
    assert kc.prove(chain.calls)


def test_a_read_of_it_further_on_is_refused():
    t = given()
    program = made(t)
    calls = [
        program.raster(t["c"], t["l"], t["o"]),
        program.copy(t["s"], t["l"], WIDTH),
        program.copy(t["o"], t["z"], WIDTH),
    ]
    with pytest.raises(kc.ChainRefused, match="writes at a table's words .*only the"):
        run_of(program).chain(calls, cores=(0, CORES))


def test_a_call_that_writes_it_by_pages_too_is_refused_where_it_is_read():
    t = given()
    program = made(t)
    run_of(program).chain([program.twice(t["c"], t["l"], t["o"])], cores=(0, CORES))
    with pytest.raises(kc.ChainRefused, match="writes pages of it too; write it one"):
        run_of(program).chain(
            [program.twice(t["c"], t["l"], t["o"]), program.copy(t["o"], t["z"], 2)],
            cores=(0, CORES),
        )
