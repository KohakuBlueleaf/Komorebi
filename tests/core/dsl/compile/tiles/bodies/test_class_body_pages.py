"""A body's writes in a chain: writes=[(t, first, pages)] a write of those pages (block
g of the output), writes=[t] anywhere (blocks of one tile, the next call waiting for
every one), a read at a table's words waiting for every block of the call before,
and what stays refused: a body that writes on mover A, a read of its pages across a
direct record (no card)."""

import pytest
from chain_fixture import CORES, TILES, WIDTH, quiet, run_of, tensor

import komorebi as km
from komorebi.chain import calls as kc

BIN = "scatter(lists, screen);"


class Entry(km.Entry):
    first: int
    pages: int


class Frame(km.Program):
    @km.handler
    def scatter(self, screen: km.Tensor, lists: km.Tensor):
        km.cxx(BIN, takes=[screen, lists], reads=[screen], writes=[lists],
               on="mover_b")  # fmt: skip

    @km.handler
    def place(self, screen: km.Tensor, lists: km.Tensor, width: int):
        first = self.rank * width
        km.cxx(BIN, takes=[screen, lists], reads=[screen],
               writes=[(lists, first, width)], on="mover_b")  # fmt: skip

    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for tile in km.Stream(a.pages(first, width)):
            results.push(tile)

    @km.handler
    def table(self, counts: km.Tensor, lists: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(self.rank, 1), write=True)
        for e in counts.pages(self.rank, 1).entries(Entry, first=0, size=2, fixed=1):
            for tile in km.Stream(lists.pages(e.first, 1)):
                results.push(tile)


@km.tile_function
def plus(a, b):
    return a + b


class Stamped(km.Program):
    """stamp: block g of out by .write (sendable direct) and lists page g by a body;
    add: b may arrive direct."""

    @km.handler
    def stamp(self, a: km.Tensor, out: km.Tensor, lists: km.Tensor):
        kept = km.tile.zeros(km.fp32)
        for tile in km.Stream(a.pages(self.rank, 1)):
            kept += tile
        out.pages(self.rank, 1).write(kept)
        km.cxx(BIN, takes=[lists], writes=[(lists, self.rank, 1)], on="mover_b")

    @km.handler(direct=("b",))
    def add(self, a: km.Tensor, b: km.Tensor, out: km.Tensor, width: int):
        first = self.rank * width
        results = km.Stream(out.pages(first, width), write=True)
        for x, y in zip(
            km.Stream(a.pages(first, width)), km.Stream(b.pages(first, width))
        ):
            results.push(plus(x, y))


class OnMoverA(km.Program):
    @km.handler
    def scatter(self, screen: km.Tensor, lists: km.Tensor):
        km.cxx(BIN, takes=[screen, lists], reads=[screen], writes=[lists],
               on="mover_a")  # fmt: skip


class Both(km.Program):
    @km.handler
    def scatter(self, screen: km.Tensor, lists: km.Tensor):
        km.cxx(BIN, takes=[screen, lists], reads=[screen],
               writes=[lists, (lists, 0, 1)], on="mover_b")  # fmt: skip


def given():
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate("slcoz")}


def made(cls, t):
    return cls(l1=1 << 20, log=quiet, screen=t["s"], lists=t["l"], a=t["s"],
               out=t["o"], counts=t["c"], chain=True)  # fmt: skip


def test_a_body_s_pages_are_block_g_of_the_call_s_output():
    t = given()
    program = made(Frame, t)
    chain = run_of(program).chain(
        [program.place(t["s"], t["l"], WIDTH), program.copy(t["l"], t["o"], WIDTH)],
        cores=(0, CORES),
    )
    place, copy = chain.calls
    assert (place.output, place.width) == ("lists (0x20000)", WIDTH)
    assert copy.input == place.output
    assert copy.ranges == (kc.Range(0, WIDTH, WIDTH),)  # its own block alone
    assert kc.prove(chain.calls)


def test_the_next_call_waits_for_every_block_of_a_body_that_writes_anywhere():
    t = given()
    program = made(Frame, t)
    chain = run_of(program).chain(
        [program.scatter(t["s"], t["l"]), program.copy(t["l"], t["o"], WIDTH)],
        cores=(0, CORES),
    )
    scatter, copy = chain.calls
    assert (scatter.output, scatter.width) == ("lists (0x20000)", 1)  # a tile a core
    assert copy.input == scatter.output and copy.reads == ()
    assert copy.ranges == (kc.Range(0, 0, CORES),)  # every core's mark
    assert kc.prove(chain.calls)


def test_a_read_at_a_table_s_words_waits_for_every_block_of_the_call_before():
    t = given()
    program = made(Frame, t)
    chain = run_of(program).chain(
        [program.copy(t["s"], t["l"], WIDTH), program.table(t["c"], t["l"], t["o"])],
        cores=(0, CORES),
    )
    copy, table = chain.calls
    assert table.input == copy.output and table.ranges == (kc.Range(0, 0, TILES),)
    assert [read.earlier for read in table.reads] == [False]  # counts: outside
    with pytest.raises(kc.ChainRefused, match="at a table's words, written by call 0"):
        run_of(program).chain(
            [
                program.copy(t["s"], t["l"], WIDTH),
                program.copy(t["s"], t["z"], WIDTH),
                program.table(t["c"], t["l"], t["o"]),
            ],
            cores=(0, CORES),
        )


def test_a_read_of_a_body_s_pages_across_a_direct_record_is_refused():
    t = given()
    program = Stamped(l1=1 << 20, log=quiet, a=t["s"], b=t["o"], out=t["o"],
                      lists=t["l"], chain=True)  # fmt: skip
    calls = [
        program.stamp(t["s"], t["o"], t["l"]),
        program.add(t["s"], t["o"], t["z"], 1),
        program.add(t["z"], km.past(t["l"]), t["c"], 1),
    ]
    assert run_of(program).chain(calls, cores=(0, CORES))  # marks tell of DRAM
    calls[1] = program.add(t["s"], km.direct(t["o"]), t["z"], 1)
    with pytest.raises(kc.ChainRefused, match="nothing it waits for tells it"):
        run_of(program).chain(calls, cores=(0, CORES))  # a direct record's do not


def test_a_body_that_writes_on_mover_a_or_both_ways_is_refused_in_a_chain():
    t = given()
    for cls, says in (
        (OnMoverA, "a body that writes in a chain runs on mover_b"),
        (Both, "lists is in writes= anywhere and by its pages: state one"),
    ):
        program = made(cls, t)
        with pytest.raises(kc.ChainRefused, match=says):
            run_of(program).chain([program.scatter(t["s"], t["l"])], cores=(0, CORES))
