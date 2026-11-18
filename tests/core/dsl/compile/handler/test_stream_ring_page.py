"""A stream's buffer page of a DRAM tensor is the tensor's aligned page size (a DRAM
read's source and destination share their low 6 bits, so slot k of a ring of 32-B
pages would be off phase): 64 B for pages of 32 and 64 B, 128 B for 96 B (no card)."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compiler import compile_class


class Pages:
    """A DRAM buffer as a tensor needs it: page size, aligned page size, pages."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000
        self.aligned_page_size = -(-page_size // 64) * 64

    def accessor_args(self):
        return [2, self.page_size]


def rows(words):
    """A DRAM uint32 tensor of 64 rows of `words` words."""
    return Tensor(Pages(4 * words), (64, words), "uint32", "row")


class Rings(km.Program):
    @km.handler
    def read(self, a: km.Tensor):
        for page in km.Stream(a.pages(0, 4)):
            km.cxx("use(page);", takes=[page], on="pack")

    @km.handler
    def taken(self, b: km.Tensor):
        whole = km.Stream(b.pages(0, 4), block=4)
        for k in range(2):
            km.cxx("use(whole, k);", takes=[whole, k])

    @km.handler
    def write(self, out: km.Tensor):
        results = km.Stream(out.pages(0, 4), write=True, block=1)
        for k in range(4):
            km.cxx("fill(results, k);", takes=[results, k])


@pytest.mark.parametrize(("words", "ring"), [(8, 64), (16, 64), (24, 128), (32, 128)])
def test_a_streams_page_is_the_dram_pages_aligned_size(words, ring):
    tensors = {"a": rows(words), "b": rows(words), "out": rows(words)}
    compiled = compile_class(Rings, {}, tensors)
    pages = {b.name: b.page_bytes for b in compiled.buffers}
    assert {"kRead0", "kWrite0"} <= set(pages) and len(pages) == 3
    assert set(pages.values()) == {ring}
