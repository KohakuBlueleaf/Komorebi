"""Buffers of a DRAM tensor in rows of small pages (no card): a broadcast's ring is the
aligned page (64 B for 32, 128 for 96), the overlay stamps' two pages each a 64-B line,
and a tile packed into a row page that is not a tile's bytes is refused by name."""

import pytest

import komorebi as km
from komorebi.device.tensor import Tensor
from komorebi.dsl.compile.records import Refused
from komorebi.dsl.compiler import compile_class


class Pages:
    """A DRAM buffer as a tensor needs it: page size, aligned page size, pages."""

    def __init__(self, page_size):
        self.page_size, self.num_pages, self.address = page_size, 64, 0x1000
        self.aligned_page_size = -(-page_size // 64) * 64

    def accessor_args(self):
        return [2, self.page_size]


def rows(words, dtype="uint32"):
    """A DRAM tensor of 64 rows of `words` words."""
    return Tensor(Pages(4 * words), (64, words), dtype, "row")


def tiles():
    """A float32 tensor of tiles."""
    return Tensor(Pages(4096), (2048, 32), "float32", "tile")


def pages_of(cls, tensors):
    """{buffer name: (page bytes, pages)} of cls built with `tensors`."""
    compiled = compile_class(cls, {}, tensors)
    return {b.name: (b.page_bytes, b.pages) for b in compiled.buffers}


class Cast(km.Program):
    @km.handler
    def go(self, t: km.Tensor):
        down = km.Broadcast(self, along="column")
        if down.sender():
            got = down.cast(t.pages(0, 4))
        else:
            got = down.arrive()
        km.cxx("use(got);", takes=[got])


@pytest.mark.parametrize(("words", "ring"), [(8, 64), (16, 64), (24, 128), (32, 128)])
def test_a_casts_ring_is_the_dram_pages_aligned_size(words, ring):
    assert pages_of(Cast, {"t": rows(words)})["kBroadcast0"][0] == ring


class Stamped(km.Program):
    @km.handler(overlay=True, stamps="s")
    def go(self, t: km.Tensor, s: km.Tensor):
        km.cxx("use(t);", takes=[t], on="mover_b")


@pytest.mark.parametrize(("words", "slot"), [(5, 64), (6, 64), (16, 64), (20, 128)])
def test_each_movers_stamps_page_starts_a_64_byte_line(words, slot):
    built = pages_of(Stamped, {"t": tiles(), "s": rows(words)})
    assert built["kStamps"] == (slot, (2,))


class Summed(km.Program):
    @km.handler
    def go(self, t: km.Tensor, out: km.Tensor):
        total = km.tile.zeros(km.fp32)
        for tile in km.Stream(t.pages(0, 4)):
            total += tile
        out.pages(0, 1).write(total)


@pytest.mark.parametrize("words", [8, 24, 512])
def test_a_tile_written_to_a_small_row_page_is_refused(words):
    with pytest.raises(Refused) as refusal:
        compile_class(Summed, {}, {"t": tiles(), "out": rows(words, "float32")})
    message = str(refusal.value)
    assert "`out.pages(0, 1).write(total)`: out is in rows of " in message
    assert f"{4 * words} B, not tiles of compute (4096 B of float32)" in message
    assert 'write its pages from a body on a mover (`km.cxx(..., on="mover_b")`' in (
        message
    )


def test_a_tile_written_to_rows_of_a_tiles_bytes_still_builds():
    built = pages_of(Summed, {"t": tiles(), "out": rows(1024, "float32")})
    assert built["kOut0"] == (4096, (2,))
