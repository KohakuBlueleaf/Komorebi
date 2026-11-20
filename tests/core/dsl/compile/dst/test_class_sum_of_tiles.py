"""`acc += tile` into DST sums of n tiles: tile j of the stream's block adds into the
sums' tile j % n (each add went into the first tile, so an all_reduce of n tiles summed
every tile into tile 0 and left the rest zero); refused by name: a block that is not a
multiple of n, and sums that leave no DST tile for the add's copy (no card)."""

import re

import pytest
from class_lines_fixture import Pages

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


def tensors():
    return {n: Tensor(Pages(4096), (32 * 8, 32 * 8), "float32", "tile") for n in "sd"}


class Sum(km.Program):
    N = km.constant(4)  # the sums' tiles
    B = km.constant(4)  # the stream's block

    @km.handler
    def go(self, s: km.Tensor, d: km.Tensor):
        acc = km.tile.zeros(km.fp32, self.N)
        for t in km.Stream(s.pages(self.rank * self.B, self.B), block=self.B):
            acc += t
        whole = km.all_reduce(acc)
        d.pages(self.rank * self.N, self.N).write(whole)


def made(n, block):
    return compile_class(Sum, {}, tensors(), constants={"N": n, "B": block})


def adds(n, block):
    compiled = made(n, block)
    text = "\n".join(render(prune(compiled.compute["go"]), 0))
    loop = text.split("km::reduce(", 1)[0]  # the stream's adds, before the reduce
    return re.findall(r"add_binary_tile\(([^,]+), (\d+), ([^)]+)\);", loop)


def test_each_tile_of_the_block_adds_into_its_own_tile_of_the_sums():
    assert adds(1, 1) == [("0", "1", "0")]
    assert adds(4, 4) == [("km_tile0", "4", "km_tile0")]
    assert adds(3, 6) == [("km_tile0 % 3", "3", "km_tile0 % 3")]


@pytest.mark.parametrize("n, block", [(4, 2), (4, 6), (2, 1)])
def test_a_block_not_a_multiple_of_the_sums_is_refused_by_name(n, block):
    with pytest.raises(Refused) as refused:
        made(n, block)
    assert (
        f"acc holds {n} tiles: tile j of t's block adds into its tile j % {n}, so "
        f"the block is a multiple of {n} tiles" in str(refused.value)
    )


class Full(km.Program):
    @km.handler
    def go(self, s: km.Tensor, d: km.Tensor):
        acc = km.tile.zeros(km.fp32, 8)
        for t in km.Stream(s.pages(0, 8), block=8):
            acc += t
        d.pages(0, 8).write(acc)


def test_sums_that_fill_dst_leave_no_tile_for_the_copy_and_are_refused():
    with pytest.raises(Refused, match="acc's 8 tiles leave no DST tile for the add"):
        compile_class(Full, {}, tensors())
