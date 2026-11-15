"""A chain call of several groups: each group's ranks share K as km::partition shares
it, the group's root (self.leader after the reduce, or `self.rank % G == 0`) writes
block g and reads its own range; the leader's if before the reduce is rank 0's
alone and refused in several groups (no card)."""

import pytest
from chain_fixture import CORES, TILES, quiet, run_of, tensor

import komorebi as km
from komorebi import chain as kc

G = 2  # a group's ranks
GROUPS = CORES // G


class Grouped(km.Program):
    """part_sum: each group's share of s's tiles up its tree, its root writing page
    g of t; rooted_sum: its root also drains page g of b first; first_only: the
    leader's write before the reduce (then each root's sum into b)."""

    @km.handler
    def part_sum(self, s: km.Tensor, t: km.Tensor, tiles: int):
        share = self.rank % G
        k0 = share * (tiles // G) + min(share, tiles % G)
        k1 = (share + 1) * (tiles // G) + min(share + 1, tiles % G)
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(k0, k1 - k0)):
            total += tile
        whole = km.reduce(total, form="tree", group=G)
        if self.rank == self.leader:
            t.pages(self.rank // G, 1).write(whole)

    @km.handler
    def rooted_sum(self, s: km.Tensor, b: km.Tensor, t: km.Tensor, tiles: int):
        share = self.rank % G
        k0 = share * (tiles // G) + min(share, tiles % G)
        k1 = (share + 1) * (tiles // G) + min(share + 1, tiles % G)
        if self.rank % G == 0:
            km.Stream(b.pages(self.rank // G, 1)).drain()
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(k0, k1 - k0)):
            total += tile
        whole = km.reduce(total, form="tree", group=G)
        if self.rank == self.leader:
            t.pages(self.rank // G, 1).write(whole)

    @km.handler
    def first_only(self, s: km.Tensor, b: km.Tensor, t: km.Tensor, tiles: int):
        if self.rank == self.leader:
            results = km.Stream(t.pages(0, 1), write=True)
            for tile in km.Stream(s.pages(0, 1)):
                results.push(tile)
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank % G, 1)):
            total += tile
        whole = km.reduce(total, form="tree", group=G)
        if self.rank == self.leader:
            b.pages(self.rank // G, 1).write(whole)


def given():
    return {n: tensor(TILES, 0x10000 * (i + 1)) for i, n in enumerate("sbt")}


def made(t):
    return Grouped(l1=1 << 20, log=quiet, s=t["s"], b=t["b"], t=t["t"], chain=True)


def test_each_groups_ranks_share_k_and_its_root_writes_block_g():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [program.part_sum(t["s"], t["t"], TILES)], cores=(0, CORES)
    )
    (call,) = chain.calls
    assert (call.group, call.size, call.width) == ((0, CORES), G, 1)
    assert call.output == "t (0x30000)" and call.sends
    assert call.reads == (kc.Read("s (0x10000)", (0, TILES), (), False),)


def test_a_read_under_rank_modulo_the_group_is_each_roots_range():
    t = given()
    program = made(t)
    chain = run_of(program).chain(
        [program.rooted_sum(t["s"], t["b"], t["t"], TILES)], cores=(0, CORES)
    )
    (call,) = chain.calls
    reads = {read.tensor: read for read in call.reads}
    assert reads["b (0x20000)"].ranges == (kc.Range(0, 1, 1, True),)
    assert reads["s (0x10000)"].k == (0, TILES)
    assert GROUPS > 1


def test_the_leaders_if_before_the_reduce_is_rank_0s_and_refused():
    t = given()
    program = made(t)
    with pytest.raises(kc.ChainRefused) as error:
        run_of(program).chain(
            [program.first_only(t["s"], t["b"], t["t"], TILES)], cores=(0, CORES)
        )
    assert "the leader alone writes t, and the call has several groups" in str(
        error.value
    )
