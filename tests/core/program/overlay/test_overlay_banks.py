"""The overlay images' DRAM layout against a simulated interleaved tensor: page p in
bank p % B at p // B pages into it. Core i reads its copy in bank i % 8 at the table's
address; B = 8 on a Blackhole card, 12 on a Wormhole chip."""

from types import SimpleNamespace

import numpy as np
import pytest

from komorebi.program import overlay
from komorebi.program.addon import code as added_code
from komorebi.program.addon import program as added_program

COPIES = overlay.constants()["kOverlayBanks"]
CORES = 120
# the parts of the two runs that hung on a Wormhole chip (overlays, two programs)
OVERLAYS = {(1, "reader"): 128, (2, "math"): 1216, (2, "pack"): 1728,
            (2, "reader"): 1024, (2, "unpack"): 1664, (2, "writer"): 1920,
            (63, "reader"): 8640}  # fmt: skip
TWO_PROGRAMS = {(number, part): size
                for number, sizes in ((62, (1984, 1600, 2624, 1664, 1728)),
                                      (61, (1984, 2048, 1728, 1216, 1664)))
                for part, size in zip(overlay.PARTS, sizes, strict=True)}  # fmt: skip
TWO_PROGRAMS[(63, "reader")] = 8640


def built_of(sizes):
    """Parts whose every word names the part (its index + 1), so a read tells which
    part it met (0: zeros)."""
    return overlay.Built(
        {
            key: np.full(size // 4, index + 1, np.uint32).tobytes()
            for index, (key, size) in enumerate(sorted(sizes.items()))
        },
        {},
    )


def read(rows, banks, bank, row, words):
    """What a read of `words` words from bank `bank` at `row` pages into it meets."""
    out = []
    while len(out) < words:
        page = row * banks + bank
        out.extend(rows[page] if page < len(rows) else np.zeros(rows.shape[1]))
        row += 1
    return np.asarray(out[:words], np.uint32)


def old_dram_rows(built, reserve=0):
    """The layout before the device's banks were asked: 8 pages a part."""
    width = max(len(image) for image in built.images.values())
    rooms = -(-reserve // width)
    rows = np.zeros((COPIES * (len(built.images) + rooms), width // 4), np.uint32)
    placed = {}
    for index, (key, image) in enumerate(sorted(built.images.items())):
        rows[COPIES * index : COPIES * (index + 1), : len(image) // 4] = np.frombuffer(
            image, np.uint32
        )
        placed[key] = (index, len(image))
    return rows, placed


def wrong_reads(layout, sizes, banks, reserve=0):
    """The (part, core) reads that meet another part or zeros."""
    built = built_of(sizes)
    rows, placed = layout(built, reserve)
    names = {key: index + 1 for index, key in enumerate(sorted(sizes))}
    return [
        (key, core)
        for key, (row, size) in placed.items()
        for core in range(CORES)
        if not (read(rows, banks, core % COPIES, row, size // 4) == names[key]).all()
    ]


@pytest.mark.parametrize("banks", [8, 12])
@pytest.mark.parametrize("sizes", [OVERLAYS, TWO_PROGRAMS])
def test_every_core_reads_each_part_it_names(banks, sizes):
    def layout(built, reserve):
        return overlay.dram_rows(built, reserve, banks)

    assert wrong_reads(layout, sizes, banks, reserve=31 * 8640) == []


@pytest.mark.parametrize("sizes", [OVERLAYS, TWO_PROGRAMS])
def test_the_old_layout_reads_wrong_on_twelve_banks(sizes):
    assert wrong_reads(old_dram_rows, sizes, 8) == []
    assert len(wrong_reads(old_dram_rows, sizes, 12)) > CORES  # the hang's cause


@pytest.mark.parametrize("sizes", [OVERLAYS, TWO_PROGRAMS])
@pytest.mark.parametrize("reserve", [0, 1000, 29 * 8640])
def test_eight_banks_lay_the_image_out_as_before(sizes, reserve):
    built = built_of(sizes)
    rows, placed = overlay.dram_rows(built, reserve, 8)
    before, placed_before = old_dram_rows(built, reserve)
    assert placed == placed_before
    assert rows.tobytes() == before.tobytes()


@pytest.mark.parametrize("banks", [8, 12])
@pytest.mark.parametrize("cores", [119, 120])
def test_every_core_reads_its_own_args_row(banks, cores):
    rows = [np.full(4, core + 1, np.uint32) for core in range(cores)]
    table = added_program.args_rows(rows, cores, 64, banks)
    for core in range(cores):
        got = read(table, banks, core % COPIES, core // COPIES, 16)
        assert (got[:4] == core + 1).all() and not got[4:].any()
    if banks == 8:  # a row a core, as before
        before = np.zeros((cores, 16), np.uint32)
        before[:, :4] = np.asarray(rows)
        assert table.tobytes() == before.tobytes()


def test_a_device_of_fewer_banks_than_the_copies_is_refused():
    with pytest.raises(ValueError, match="a device of 6 DRAM banks"):
        overlay.dram_rows(built_of(OVERLAYS), 0, 6)


@pytest.mark.parametrize("banks", [8, 12])
def test_an_added_part_of_several_rows_is_read_whole_by_every_core(banks, monkeypatch):
    width, first = 256, 2  # the room's rows from 2; a part over 3 rows, one over 1
    images = [np.full(150, 7, np.uint32).tobytes(), np.full(40, 9, np.uint32).tobytes()]
    rows_of = {images[0]: first, images[1]: first + 3}
    monkeypatch.setattr(
        added_code, "_rows", lambda room, made, builds: (rows_of, first + 4, images)
    )
    memory = np.zeros((banks * (first + 4), width), np.uint8)

    def write(tensor, pages, first):
        memory[first : first + len(pages)] = pages

    image = SimpleNamespace(address=0x10000, aligned_page_bytes=width)
    room = SimpleNamespace(row_bytes=width, next=first, image=image)
    card = SimpleNamespace(dram_banks=banks, write=write)
    run = SimpleNamespace(backend=SimpleNamespace(room=room, card=card))
    placed = added_code._place(run, None, [{"reader": images[0], "math": images[1]}])
    words = memory.view(np.uint32)
    for part, value in (("reader", 7), ("math", 9)):
        address, size = placed[0][part]
        row = (address - image.address) // width
        for core in range(CORES):
            got = read(words, banks, core % COPIES, row, size // 4)
            assert (got == value).all(), (part, core)
