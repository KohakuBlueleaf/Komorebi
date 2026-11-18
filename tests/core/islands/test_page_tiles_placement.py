"""A tile tensor in DRAM pages of k tiles (page_tiles) on the host: its page and
buffer bytes, its pages placed over islands in steps of k-tile pages, each island's
part of the same page bytes; and each refusal: not a tile layout, a width not whole
pages, a page past one NoC packet, an L1 tensor, a read a tile a page (no card)."""

from types import SimpleNamespace

import pytest

from komorebi import formats
from komorebi.device import tensor as tensor_module
from komorebi.islands import graph as islands
from komorebi.islands.graph import Island, Link
from komorebi.islands.parts.entries import EntriesMixin
from komorebi.session.parts.transfer import TransferMixin

TWO = islands.Graph([Island(tuple(range(120)), 12)] * 2, [Link(0, 1)])


def test_a_page_of_k_tiles_and_the_same_bytes_in_all():
    shape = (64, 256)  # 2 x 8 tiles
    assert formats.page_bytes(shape, "bfp8_b", "tile", 4) == 4 * 1088
    assert formats.page_bytes(shape, "bfp4_b", "tile", 8) == 8 * 576
    assert formats.buffer_bytes(shape, "bfp8_b", "tile", 4) == formats.buffer_bytes(
        shape, "bfp8_b", "tile"
    )
    assert formats.page_bytes(shape, "bfp8_b", "tile") == 1088


def test_the_page_refusals_are_named():
    with pytest.raises(ValueError, match="page_tiles=2 is for a tile layout"):
        formats.page_bytes((64, 256), "uint32", "row", 2)
    with pytest.raises(ValueError, match="is 9 tiles, not a multiple of 4"):
        formats.page_bytes((64, 288), "bfp8_b", "tile", 4)
    with pytest.raises(ValueError, match="16,384 B .at most 8 tiles of bfloat16"):
        formats.page_bytes((64, 512), "bfloat16", "tile", 16)
    with pytest.raises(ValueError, match="page_tiles=0: a whole number of tiles"):
        formats.page_bytes((64, 256), "bfp8_b", "tile", 0)
    assert formats.page_bytes((64, 32 * 15), "bfp8_b", "tile", 15) == 15 * 1088


def test_an_l1_tensor_in_pages_of_tiles_is_refused():
    with pytest.raises(ValueError, match="page_tiles=2 with shard_grid"):
        tensor_module.empty(None, None, (64, 256), "bfp8_b", "tile", (2, 1), None, 2)


def test_a_split_along_a_row_is_in_pages_of_k_tiles():
    edge = TWO.first(1)
    one = islands.place(TWO, (32, 168 * 32), "tile", -1, (edge - 84, 168))
    seven = islands.place(TWO, (32, 168 * 32), "tile", -1, (edge - 12, 24), False, 7)
    assert islands.page_dims((32, 168 * 32), "tile", 7) == (1, 24)
    assert one.pages == 168 and seven.pages == 24
    assert [one.part_pages(i) for i in (0, 1)] == [84, 84]
    assert [seven.part_pages(i) for i in (0, 1)] == [12, 12]
    assert seven.where(12) == (1, 0)  # tiles 84 .. 90: island 1's first page


class Entries(EntriesMixin):
    """An island session's tensor entries over TWO, no card: parts are stand-ins."""

    def __init__(self):
        self.graph, self.card_dram, self.split = TWO, False, 1
        self.sessions = (SimpleNamespace(), SimpleNamespace())
        self.made = []
        self._init_entries(4)
        self._tables = []  # no load's table yet

    def log(self, line):
        pass

    def _table_row(self, word):
        pass


class Part:
    """One island's part as a session's empty() makes it."""

    def __init__(self, shape, dtype, layout, page_tiles=1):
        self.shape, self.page_tiles = shape, page_tiles
        self.page_bytes = formats.page_bytes(shape, dtype, layout, page_tiles)
        self.aligned_page_bytes = self.page_bytes
        self.address = 0x10000

    def free(self):
        pass


def test_each_island_s_part_holds_its_pages_of_k_tiles():
    entries = Entries()
    made = []

    def empty(shape, dtype, layout, **paged):
        made.append((shape, paged))
        return Part(shape, dtype, layout, **paged)

    for session in entries.sessions:
        session.empty = empty
    tensor = entries.empty(
        (5376, 168 * 32), "bfp8_b", "tile", -1, (TWO.first(1) - 12, 24), page_tiles=7
    )
    assert made == [((168 * 12 * 32, 7 * 32), {"page_tiles": 7})] * 2
    assert tensor.page_bytes == 7 * 1088 and tensor.page_tiles == 7
    with pytest.raises(ValueError, match="is 168 tiles, not a multiple of 5"):
        entries.empty((5376, 168 * 32), "bfp8_b", "tile", -1, page_tiles=5)


def test_a_read_a_tile_a_page_refuses_pages_of_k_tiles():
    held = SimpleNamespace(layout="tile", dtype="bfloat16", page_tiles=2)
    with pytest.raises(ValueError, match="this tensor is in pages of 2 tiles"):
        TransferMixin.read_tiles_bf16(None, held, 0, 1)
