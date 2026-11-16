"""A chain call whose groups' reads are not base + step x index: its marks wait for
the narrowest such range that holds every group's evaluated one; a direct input read
so is refused, and so is a read no range word holds (no card)."""

import pytest
from chain_fixture import CORES, quiet, run_of, tensor

import komorebi as km
from komorebi.chain import calls as kc
from komorebi.dsl import chain_calls


class Clamped(km.Program):
    """own: core g writes page g of t; tail: core g reads t from page g, two pages
    but one on the last core; head: core g reads two pages from min(g, CORES - 2)."""

    @km.handler
    def own(self, s: km.Tensor, t: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            total += tile
        t.pages(self.rank, 1).write(total)

    @km.handler
    def tail(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.pages(self.rank, min(2, CORES - self.rank))):
            total += tile
        y.pages(self.rank, 1).write(total)

    @km.handler
    def head(self, t: km.Tensor, y: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.pages(min(self.rank, CORES - 2), 2)):
            total += tile
        y.pages(self.rank, 1).write(total)


def made():
    t = {n: tensor(CORES, 0x10000 * (i + 1)) for i, n in enumerate("sty")}
    program = Clamped(l1=1 << 20, log=quiet, chain=True, **t)
    return program, t


@pytest.mark.parametrize(
    ("name", "cover"), [("tail", kc.Range(0, 1, 2)), ("head", kc.Range(0, 0, CORES))]
)
def test_a_read_that_is_not_affine_waits_for_the_range_that_holds_it(name, cover):
    program, t = made()
    calls = [program.own(t["s"], t["t"]), getattr(program, name)(t["t"], t["y"])]
    chain = run_of(program).chain(calls, cores=(0, CORES))
    assert chain.calls[1].ranges == (cover,)


def test_the_cover_is_the_narrowest_and_a_range_word_must_hold_it():
    access = type("Access", (), {"line": 7, "tensor": "t"})()
    firsts = [(0, 2), (1, 2), (2, 2), (3, 1)]
    assert chain_calls.covering_range(firsts, access) == kc.Range(0, 1, 2)
    assert chain_calls.covering_range([(4, 2), (4, 2), (9, 1)], access) == (
        kc.Range(4, 0, 6)
    )
    with pytest.raises(chain_calls.Undescribed, match="line 7: t's groups read"):
        chain_calls.covering_range([(0, 2000), (5000, 2)], access)
