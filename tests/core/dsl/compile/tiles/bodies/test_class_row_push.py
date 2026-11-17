"""A push of a stream page of a tensor in rows whose page is not a tile's bytes (uint32
rows of 8 words and the like) is refused by name: copy_tile and pack_tile move a whole
tile. Tiles, rows of a tile's bytes and a body on a mover still build."""

import pytest

import komorebi as km
from komorebi import formats
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class

MOVED = "noc_async_read_page(first, km::accessor_of(a), get_write_ptr(stage));"


class Pages:
    """A DRAM buffer as a tensor needs it: page size, pages, address."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000

    def accessor_args(self):
        return [2, self.page_size]


def rows(words):
    """A uint32 tensor of 64 rows of `words` words."""
    return Tensor(Pages(4 * words), (64, words), "uint32", "row")


def tiles():
    """A bfloat16 tensor of tiles."""
    return Tensor(Pages(formats.tile_bytes("bfloat16")), (2048, 32), "bfloat16", "tile")


class Copy(km.Program):
    @km.handler
    def copy(self, a: km.Tensor, out: km.Tensor):
        results = km.Stream(out.pages(0, 4), write=True)
        for page in km.Stream(a.pages(0, 4)):
            results.push(page)


@pytest.mark.parametrize("words", [8, 16, 24])
def test_a_push_of_a_page_in_rows_is_refused(words):
    with pytest.raises(Refused) as refusal:
        compile_class(Copy, {}, {"a": rows(words), "out": rows(words)})
    message = str(refusal.value)
    assert f"`results.push(page)`: page is a page of a, in rows of {4 * words} B, " in (
        message
    )
    assert "not a tile of compute (4096 B of uint32)" in message
    assert "a push copies and packs it as a whole tile" in message
    assert 'move its pages on a mover in a body (`km.cxx(..., on="mover_b")`' in (
        message
    )


def test_a_push_of_a_tile_still_builds():
    compile_class(Copy, {}, {"a": tiles(), "out": tiles()})


def test_a_push_of_rows_of_a_tiles_bytes_still_builds():
    compile_class(Copy, {}, {"a": rows(1024), "out": rows(1024)})


class Moved(km.Program):
    stage: km.L1[km.u32, 1, 32, 32] = km.own(push=None)

    @km.handler
    def moved(self, a: km.Tensor, out: km.Tensor, first: int):
        km.cxx(MOVED, takes=[a, out, first, self.stage], reads=[a], writes=[out],
               on="mover_b")  # fmt: skip


def test_a_body_on_a_mover_still_moves_the_rows():
    compile_class(Moved, {}, {"a": rows(8), "out": rows(8)})
