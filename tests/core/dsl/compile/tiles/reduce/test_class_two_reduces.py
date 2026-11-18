"""Two reduces in one handler: of one form and group they build, the second a turn
after the first (its senders wait for their receiver's credit); another form or
another group size is refused; a chain describes the handler (no card)."""

import pytest
from chain_fixture import CORES, TILES, quiet, run_of, tensor

import komorebi as km
from komorebi.dsl.compile.records import Refused


class Two(km.Program):
    """Two sums of s's pages up the command's tree, each written by the leader."""

    @km.handler
    def sums(self, s: km.Tensor, t: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank + 8, 1)):
            other += tile
        second = km.reduce(other)
        if self.rank == self.leader:
            t.pages(1, 1).write(second)


class Grouped(km.Program):
    """Sums of s and of z in groups of 2, a share of the pages a rank (as
    km::partition shares them), each group's root writing page g of t and of y."""

    @km.handler
    def sums(self, s: km.Tensor, z: km.Tensor, t: km.Tensor, y: km.Tensor, tiles: int):
        share = self.rank % 2
        k0 = share * (tiles // 2) + min(share, tiles % 2)
        k1 = (share + 1) * (tiles // 2) + min(share + 1, tiles % 2)
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(k0, k1 - k0)):
            mine += tile
        whole = km.reduce(mine, form="tree", group=2)
        if self.rank == self.leader:
            t.pages(self.rank // 2, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(z.pages(k0, k1 - k0)):
            other += tile
        second = km.reduce(other, form="tree", group=2)
        if self.rank == self.leader:
            y.pages(self.rank // 2, 1).write(second)


class OtherForm(km.Program):
    @km.handler
    def sums(self, s: km.Tensor, t: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine, group=2)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            other += tile
        second = km.reduce(other, form="owner", group=2)  # owner
        if self.rank == self.leader:
            t.pages(1, 1).write(second)


class OtherGroup(km.Program):
    @km.handler
    def sums(self, s: km.Tensor, t: km.Tensor):
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            mine += tile
        whole = km.reduce(mine, group=2)
        if self.rank == self.leader:
            t.pages(0, 1).write(whole)
        other = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            other += tile
        second = km.reduce(other, group=4)  # four
        if self.rank == self.leader:
            t.pages(1, 1).write(second)


GIVEN = {
    "s": tensor(TILES, 0x10000),
    "t": tensor(TILES, 0x20000),
    "y": tensor(TILES, 0x30000),
    "z": tensor(TILES, 0x40000),
}


def made(cls, chain=False):
    return cls(l1=1 << 20, log=quiet, chain=chain, **GIVEN)


def test_two_reduces_in_groups_of_one_size_build():
    program = made(Grouped)
    size, line = program.reduce_groups["sums"]
    assert size == 2 and "group=2" in line


@pytest.mark.parametrize(
    ("cls", "first"),
    [(OtherForm, "reduces by tree in groups of 2"), (OtherGroup, "in groups of 2")],
)
def test_another_form_or_group_is_refused_with_what_to_write(cls, first):
    with pytest.raises(Refused) as error:
        made(cls)
    text = str(error.value)
    assert "the reduces of a handler share one form and group" in text
    assert first in text and "whole = km.reduce(mine, group=2)" in text
    assert "would send before its receiver frees the slot" in text
    assert "or reduce in a handler of its own (two calls in a chain)" in text


def test_a_chain_describes_a_handler_with_two_reduces():
    program = made(Grouped, chain=True)
    s, z, t, y = (GIVEN[n] for n in "szty")
    chain = run_of(program).chain([program.sums(s, z, t, y, TILES)], cores=(0, CORES))
    (call,) = chain.calls
    assert call.size == 2 and call.sends
