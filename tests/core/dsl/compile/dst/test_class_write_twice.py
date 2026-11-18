"""A tile of DST or a reduce's sum written to two tensors: compute packs it once and
mover B writes its pages twice, freeing them after the last; on the two branches of
an if (a command word, or nested under the leader's if) each branch writes it; on one
branch alone, the other empty, it is freed there; the refusals (no card)."""

import pytest
from chain_fixture import CORES, TILES, quiet, run_of, tensor

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render

GIVEN = {
    "s": tensor(TILES, 0x10000),
    "t": tensor(TILES, 0x20000),
    "y": tensor(TILES, 0x30000),
}


class Twice(km.Program):
    """tile: each rank's sum to t and y; summed: the leader's reduce to t and y."""

    @km.handler
    def tile(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        t.pages(self.rank, 1).write(mine)
        y.pages(self.rank, 1).write(mine)

    @km.handler
    def summed(self, s: km.Tensor, t: km.Tensor, y: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
            y.pages(0, 1).write(whole)


class Branches(km.Program):
    """word: the tile to t or y by a command word; leader: the reduce's sum so, under
    the leader's if; only: to t when the word is 0, else nothing."""

    @km.handler
    def word(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        if which == 0:
            t.pages(self.rank, 1).write(mine)
        else:
            y.pages(self.rank, 1).write(mine)

    @km.handler
    def leader(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            if which == 0:
                t.pages(0, 1).write(whole)
            else:
                y.pages(0, 1).write(whole)

    @km.handler
    def only(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:  # noqa: SIM102  (the leader's, then the word's)
            if which == 0:
                t.pages(0, 1).write(whole)


class UsedAfter(km.Program):
    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        if which == 0:
            t.pages(self.rank, 1).write(mine)
        y.pages(self.rank, 1).write(mine)  # after


class OtherLines(km.Program):
    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        if which == 0:  # lines
            t.pages(self.rank, 1).write(mine)
        else:
            km.Stream(s.pages(0, 1)).drain()


class OtherList(km.Program):
    @km.handler
    def go(self, s: km.Tensor, t: km.Tensor, y: km.Tensor, which: int):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
        if self.rank == self.leader:
            y.pages(0, 1).write(whole)  # another list


def made(cls, chain=False):
    return cls(l1=1 << 20, log=quiet, chain=chain, **GIVEN)


def text(cls, kernel, handler):
    compiled = compile_class(cls, {}, GIVEN)
    return "\n".join(render(prune(getattr(compiled, kernel)[handler]), 0))


def test_two_writes_pack_once_and_free_after_the_last():
    for handler in ("tile", "summed"):
        movement = text(Twice, "movement", handler)
        # one page held for two writes: a pack each would give the same values
        assert movement.count("cb_pop_front(kOut0, 1);") == 1
    assert text(Twice, "compute", "tile").count("pack_tile(") == 1


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        (UsedAfter, "mine is written on one branch of the if at line"),
        (OtherLines, "mine is written on one branch of this if only, and the other"),
        (OtherList, "whole is written at line"),
    ],
)
def test_the_refusals_name_the_branch_or_the_list(cls, message):
    with pytest.raises(Refused, match=message):
        made(cls)


def test_a_chain_describes_two_outputs_of_one_tile():
    program = made(Twice, chain=True)
    s, t, y = (GIVEN[n] for n in "sty")
    chain = run_of(program).chain([program.summed(s, t, y)], cores=(0, CORES))
    (call,) = chain.calls
    assert call.output == "t (0x20000)"
    assert [str(o) for o, _ in call.outputs] == ["y (0x30000)"]


def test_a_chain_refuses_a_write_on_a_words_branch_by_name():
    program = made(Branches, chain=True)
    s, t, y = (GIVEN[n] for n in "sty")
    with pytest.raises(kc.ChainRefused, match="a read or write under an if on"):
        run_of(program).chain([program.word(s, t, y, 0)], cores=(0, CORES))
