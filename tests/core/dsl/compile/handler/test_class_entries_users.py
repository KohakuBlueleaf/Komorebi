"""A table of entries read only where its words are used: on mover B alone when only
mover B's lines use them, on every CPU when compute's do (no card)."""

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class
from komorebi.dsl.front.nodes import prune, render


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size, pages):
        self.page_size, self.num_pages, self.address = page_size, pages, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


class Cut(km.Entry):
    tile: int


class Tables(km.Program):
    @km.handler
    def writer(self, heavy: km.Tensor):
        for cut in heavy.pages(0, 1).entries(Cut, first=1, size=1, count=0):
            km.cxx("write(cut_tile);", takes=[cut], on="mover_b")

    @km.handler
    def everyone(self, heavy: km.Tensor):
        for cut in heavy.pages(0, 1).entries(Cut, first=1, size=1, count=0):
            km.cxx("count(cut_tile);", takes=[cut], on="pack")


def kernels(handler):
    tensors = {"heavy": Tensor(Pages(4096, 1), (1, 1024), "uint32", "row")}
    compiled = compile_class(Tables, {}, tensors)
    return [
        "\n".join(render(prune(nodes[handler]), 0))
        for nodes in (compiled.movement, compiled.compute)
    ]


def test_a_table_only_mover_b_uses_is_read_on_mover_b_alone():
    movement, compute = kernels("writer")
    assert "COMPILE_FOR_BRISC)  // mover A" not in movement
    assert "km::read_pages" in movement
    assert "wait_front" not in compute


def test_a_table_compute_uses_is_read_for_compute_as_before():
    movement, compute = kernels("everyone")
    assert "// mover A" in movement
    assert "compute.wait_front(" in compute
