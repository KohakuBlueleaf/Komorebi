"""`whole = km.reduce(acc, ...)` read by a later stream loop of the same handler: the
sums written to a hidden fp32 field, all-reduced so every rank holds them, and the
loop's tile function reads that field where the source says `whole` (no card)."""

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.pages, self.address = page_size, 64, 0x9000


@km.tile_function
def scaled(x, w):
    return x * w.col()


class LaterLoop(km.Program):
    @km.handler
    def go(self, s: km.Tensor, y: km.Tensor):
        acc = km.tile.zeros(km.fp32)
        for tile in km.Stream(s.pages(self.rank, 1)):
            acc += tile
        whole = km.reduce(acc, form="tree", group=4)
        results = km.Stream(y.pages(self.rank, 1), write=True)
        for xt in km.Stream(s.pages(self.rank, 1)):
            results.push(scaled(xt, whole))


def test_the_loop_reads_the_all_reduced_field():
    page = formats.tile_bytes("float32")
    tensors = {n: Tensor(Pages(page), (2048, 32), "float32", "tile") for n in "sy"}
    compiled = compile_class(LaterLoop, {}, tensors)
    assert "km_whole_whole" in compiled.fields
    text = "\n".join(render(prune(compiled.compute["go"]), 0))
    taken = text.split("km::all_reduce_take(", 1)
    assert len(taken) == 2  # the field all-reduced: every rank holds it
    assert "kFieldKmWholeWhole" in taken[1].split("});", 1)[1]  # read after it
