"""A DRAM tensor that does not fit what is left of one card's DRAM is refused by name
before tt-metal's allocator is asked: the tensor, its bytes, what each bank has left;
an island's part refused so names the island (no card: the allocator's figures faked).
"""

import pytest

from komorebi.device import tensor as tensor_module
from komorebi.islands.parts.entries import EntriesMixin

BANKS, ALIGN = 8, 64
SHAPE = (4096, 32768)  # 256 MiB of bfloat16 tiles: 131,072 pages of 2,048 bytes


class Device:
    """The native device's dram_room: tt-metal's spread over its banks, and a bank's
    largest free block and free bytes as given."""

    def __init__(self, largest):
        self.largest = largest

    def dram_room(self, size, page):
        pages = size // page
        bank = -(-pages // BANKS) * (-(-page // ALIGN) * ALIGN)
        return bank, BANKS, self.largest, self.largest + 4096


class Module:
    def __init__(self):
        self.asked = []

    def Buffer(self, device, size, page, shard_grid, address):
        self.asked.append(size)
        return type("Buffer", (), {"page_size": page, "num_pages": size // page})()


def test_a_tensor_past_the_card_s_dram_is_refused_before_the_allocator():
    module = Module()
    tensor = tensor_module.empty(module, Device(32 << 20), SHAPE, "bfloat16", "tile")
    assert module.asked == [256 << 20] and tensor.shape == SHAPE
    with pytest.raises(tensor_module.DramFull) as refused:
        tensor_module.empty(module, Device(20 << 20), SHAPE, "bfloat16", "tile")
    assert module.asked == [256 << 20]  # tt-metal was not asked
    said = str(refused.value)
    assert said.startswith("a DRAM tensor of shape (4096, 32768) (bfloat16, tile), ")
    assert "268,435,456 bytes, does not fit what is left of the card's DRAM" in said
    assert "33,554,432 bytes in each of its 8 banks" in said
    assert "20,971,520 free in one piece (20,975,616 free in all)" in said
    assert "at most 81,920 pages of 2,048 bytes fit now" in said
    assert refused.value.fits == 81920


def test_an_island_s_part_refused_so_names_the_island_and_what_fits():
    class Session:
        def empty(self, shape, dtype, layout):
            return tensor_module.empty(Module(), Device(64 << 10), shape, dtype, layout)

    placement = type("Placement", (), {"pages": 1024, "part_pages": lambda s, i: 512})()
    with pytest.raises(tensor_module.KomorebiError) as refused:
        EntriesMixin()._part(
            Session(), 1, placement, (1024 * 32, 32), "bfloat16", "tile"
        )
    said = str(refused.value)
    assert said.startswith("island 1: its part of 512 pages of shape (32768, 32) ")
    assert (
        "does not fit its DRAM (131,072 bytes a bank asked, 65,536 free in one" in said
    )
    assert "at most 256 pages of this kind fit there now" in said
