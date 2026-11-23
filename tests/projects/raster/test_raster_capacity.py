"""The frame's capacity in tiles (card_thread.capacity): the sizes the kernels' setup.h
and the thread's frame.h compute from it, built on the host, against the Python
mirrors (card_passes.sizes, card_thread.thread_sizes, work_bytes); the capacities of
the sizes people use; the frame tensors' chunks."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from projects.raster import card, card_passes, card_thread, lists

KERNELS = lists.HERE / "kernels"
THREAD = lists.HERE / "thread"
CAPACITIES = (32, 928, lists.TILES, 2048, 3616, 8160)
CORES = 120  # the fill cores of a Blackhole card's frame program (session.ranks)
SETUP_NAMES = (
    "kCountBytes",
    "kHistBytes",
    "kSlotsBytes",
    "kItemsBytes",
    "kItemRows",
    "kBaseAt",
    "kHprefixAddrAt",
    "kShadeAddrAt",
    "kFrameWordsAt",
    "kItemWordsAt",
    "kCoreWordsAt",
    "kItemsRegionBytes",
    "kSumsHeavyAt",
)
FRAME_NAMES = (
    "RF_COUNTER_BYTES",
    "RF_SLOTS_BYTES",
    "RF_ITEMS_BYTES",
    "RF_DISPLAY_PAGES",
    "RF_BASE_AT",
    "RF_HPREFIX_ADDR_AT",
    "RF_SHADE_ADDR_AT",
    "RF_SUMS_HEAVY_AT",
    "RF_SUMS_BYTES",
    "RF_FRAME_WORDS_AT",
    "RF_ITEM_WORDS_AT",
    "RF_CORE_WORDS_AT",
    "RF_ITEMS_REGION_BYTES",
)


def held_of(tiles, recount=False, cores=CORES):
    return card_thread.Capacity(tiles, True, recount, cores)


@pytest.mark.parametrize(
    ("tiles", "recount", "cores"),
    [(tiles, False, CORES) for tiles in CAPACITIES]
    + [
        (3616, True, CORES),
        (8160, True, CORES),
        (8160, True, 140),
        (lists.TILES, 0, None),
    ],
)
def test_setup_sizes(tmp_path, tiles, recount, cores):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("needs a C++ compiler (g++)")
    prints = " ".join(f'printf("%u\\n", (unsigned){name});' for name in SETUP_NAMES)
    source = f'#include <cstdio>\n#include "setup.h"\nint main() {{ {prints} }}\n'
    held = held_of(tiles, bool(recount), cores)
    heavy, band_items = card.heavy_capacity(held)
    flags = ["-std=c++17", f"-I{KERNELS}", f"-DRASTER_TILES={tiles}"]
    flags += [f"-DRASTER_HEAVY={heavy}", f"-DRASTER_BAND_ITEMS={band_items}"]
    flags += ["-DRASTER_RECOUNT=1"] if recount else []
    path = tmp_path / "setup.cpp"
    path.write_text(source)
    out = tmp_path / "setup"
    subprocess.run([compiler, *flags, str(path), "-o", str(out)], check=True)
    run = subprocess.run([str(out)], check=True, capture_output=True, text=True)
    got = dict(zip(SETUP_NAMES, (int(n) for n in run.stdout.split()), strict=True))
    want = card_passes.sizes(tiles, bool(recount), heavy, band_items)
    assert got == {name: want[name] for name in SETUP_NAMES}


@pytest.mark.parametrize("tiles", CAPACITIES)
def test_thread_sizes(tmp_path, tiles):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("needs a C compiler (cc)")
    prints = " ".join(f'printf("%u\\n", (unsigned){name});' for name in FRAME_NAMES)
    source = (
        "#include <stdio.h>\n#include <stdint.h>\n"
        '#include "middle.h"\n#include "frame.h"\n'
        "typedef struct {\n"
        "    rm_scratch scratch;\n"
        "    rm_item items[RM_MAX_ITEMS];\n"
        "    uint32_t tile_item[RM_TILES + 1];\n"
        "    uint32_t heavy[RM_MAX_HEAVY], bands[RM_MAX_HEAVY];\n"
        "    uint32_t ranges[RF_AREAS][RF_MAX_CORES + 1];\n"
        "    uint32_t cuts[RF_MAX_CORES + 1];\n"
        "} work_t;\n"
        f'int main(void) {{ {prints} printf("%u\\n", (unsigned)sizeof(work_t)); }}\n'
    )
    path = tmp_path / "frame.c"
    path.write_text(source)
    out = tmp_path / "frame"
    held = held_of(tiles)
    heavy, band_items = card.heavy_capacity(held)
    flags = ["-std=c11", f"-I{THREAD}", f"-DRM_TILES={tiles}u"]
    flags += [f"-DRM_MAX_HEAVY={heavy}u", f"-DRM_BAND_ITEMS={band_items}u"]
    subprocess.run([compiler, *flags, str(path), "-o", str(out)], check=True)
    run = subprocess.run([str(out)], check=True, capture_output=True, text=True)
    numbers = [int(n) for n in run.stdout.split()]
    got = dict(zip(FRAME_NAMES, numbers[:-1], strict=True))
    want = card_thread.thread_sizes(held)
    assert got == {name: want[name] for name in FRAME_NAMES}
    assert numbers[-1] <= card_thread.work_bytes(held)  # frame.c's work_t fits


def test_work_t_mirrors_frame_c():
    """The test's copy of frame.c's work_t is frame.c's."""
    text = (THREAD / "frame.c").read_text()
    for line in (
        "rm_scratch scratch;",
        "rm_item items[RM_MAX_ITEMS];",
        "uint32_t tile_item[RM_TILES + 1];",
        "uint32_t heavy[RM_MAX_HEAVY], bands[RM_MAX_HEAVY];",
        "uint32_t ranges[RF_AREAS][RF_MAX_CORES + 1];",
        "uint32_t cuts[RF_MAX_CORES + 1];",
    ):
        assert line in text


def test_capacities():
    capacity, held = card_thread.capacity, card_thread.Capacity
    assert capacity([(1024, 1024)]) == held(lists.TILES, False)
    assert capacity([(1280, 720)]) == held(928, False)
    assert capacity([(1280, 720)], lit=True) == held(lists.TILES, False)  # SUN's
    assert capacity([(1920, 1080)]) == held(2048, True)
    assert capacity([(2560, 1440)]) == held(3616, True)
    assert capacity([(3840, 2160)]) == held(8160, True)
    assert capacity([(2048, 1024)]) == held(2048, False)  # the narrow words' most
    assert capacity([(32, 32)], at_least=held(2048, True)) == held(2048, True)
    assert card_thread.covers(held(2048, True), (1280, 720))
    assert not card_thread.covers(held(2048, False), (2080, 32))  # 65 tiles a row
    assert card_thread.default_capacity() == held(lists.TILES, False)


def test_frame_chunks():
    most = lists.K["kDirFrameMax"]
    for tiles in CAPACITIES:
        pages = card_thread.PAGES_PER_ITEM * tiles
        chunk = card.chunk_pages(pages)
        assert chunk % 15 == 0 and chunk >= card.CHUNK_PAGES
        assert -(-pages // chunk) <= most
    assert (
        card.chunk_pages(card_thread.PAGES_PER_ITEM * lists.TILES) == card.CHUNK_PAGES
    )


def test_numbers_agree():
    for tiles, wide in ((lists.TILES, False), (2048, True), (8160, True)):
        held = card_thread.Capacity(tiles, wide)
        assert card_thread.same_numbers(held) == {}
        assert card_thread.same_numbers(held._replace(cores=CORES)) == {}


def test_heavy_capacity():
    """From the fill cores: 120 cores cut fewer than 120 / 0.33 = 363.6 tiles; the base
    word's 9-bit heavy index refuses past 168 cores; the host-stepped passes' 256."""
    assert card.heavy_capacity(held_of(8160)) == (364, 384)
    assert card.heavy_capacity(held_of(32)) == (32, 384)  # at most the tiles
    assert card.heavy_capacity(None) == (256, 1024)
    assert card.heavy_capacity(held_of(8160, cores=168))[0] == 510
    with pytest.raises(ValueError, match="heavy index holds 511"):
        card.heavy_capacity(held_of(8160, cores=169))


def build_middle(tmp_path, tiles, heavy, band_items):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("needs a C compiler (cc)")
    out = tmp_path / f"middle_{heavy}.so"
    command = [compiler, "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC"]
    command += [f"-DRM_TILES={tiles}u", f"-DRM_MAX_HEAVY={heavy}u"]
    command += [f"-DRM_BAND_ITEMS={band_items}u"]
    subprocess.run([*command, str(THREAD / "middle.c"), "-o", str(out)], check=True)
    library = ctypes.CDLL(str(out))
    void, word = ctypes.c_void_p, ctypes.c_uint32
    library.rm_heavy.argtypes = [void, word, word, void, void, void]
    library.rm_heavy.restype = word
    library.rm_middle.argtypes = [void, void, word, word, void, void, word]
    library.rm_middle.argtypes += [*[void] * 8, word]
    library.rm_middle.restype = word
    return library


def middle_of(library, tiles, heavy_most, band_items, entries):
    """rm_heavy and rm_middle (thread/middle.c) on one counting core's `entries` a
    tile, CORES fill cores: (heavy tiles found, items, 0 when refused, directory)."""
    counters = np.ascontiguousarray(
        np.concatenate([entries, np.zeros_like(entries)])[None], np.uint32
    )
    scratch = np.zeros(16 << 20, np.uint8)
    heavy = np.zeros(max(heavy_most, 1024), np.uint32)
    bands = np.zeros_like(heavy)
    found = library.rm_heavy(
        counters.ctypes.data, 1, CORES, scratch.ctypes.data, heavy.ctypes.data,
        bands.ctypes.data,
    )  # fmt: skip
    taken = min(found, heavy_most)
    hist = np.zeros((1, taken, 32), np.uint16)
    items_most = tiles + band_items
    items = np.zeros((items_most, 4), np.uint32)
    tile_item = np.zeros(tiles + 1, np.uint32)
    directory = np.zeros((CORES, lists.K["kDirBytes"] // 4), np.uint32)
    marks = np.zeros((items_most + 248, 6), np.uint32)
    next_slot = np.zeros((1, items_most), np.uint32)
    item_total, mark_total = ctypes.c_uint32(), ctypes.c_uint32()
    slots = library.rm_middle(
        counters.ctypes.data, hist.ctypes.data, 1, CORES, heavy.ctypes.data,
        bands.ctypes.data, found, scratch.ctypes.data, items.ctypes.data,
        ctypes.addressof(item_total), tile_item.ctypes.data, directory.ctypes.data,
        marks.ctypes.data, ctypes.addressof(mark_total), next_slot.ctypes.data,
        items_most,
    )  # fmt: skip
    return found, item_total.value, slots, directory


def test_more_than_256_heavy_tiles(tmp_path):
    """A frame of 8,160 tiles whose load is in 301 tiles (300 just above a third of a
    core's mean, one above the mean): the 256-tile table refuses it (frame.c then
    returns RF_TOO_MANY_HEAVY, the frame dropped with word 4); the table of 120 fill
    cores' capacity (364) cuts all 301 and lays out every item."""
    tiles = 8160
    entries = np.zeros(tiles, np.int64)
    entries[:300] = 3820  # 260 x 3,820 + 6,800 = 1.0 M a tile
    entries[300] = 38440  # 10 M
    stepped = build_middle(tmp_path, tiles, card.STEPPED_HEAVY, card.STEPPED_BAND_ITEMS)
    found, _, slots, _ = middle_of(
        stepped, tiles, card.STEPPED_HEAVY, card.STEPPED_BAND_ITEMS, entries
    )
    assert found == 301 and slots == 0
    heavy, band_items = card.heavy_capacity(held_of(tiles))
    sized = build_middle(tmp_path, tiles, heavy, band_items)
    found, item_count, slots, _ = middle_of(sized, tiles, heavy, band_items, entries)
    assert found == 301 and slots > 0
    assert tiles < item_count <= tiles + band_items


def test_directory_page_full(tmp_path):
    """356 tiles of 4 M (none cut: below a third of a core's mean) and 7,804 empty
    ones on 120 cores: four cores get two of the 4 M tiles, and the empty tiles, the
    least loaded first, would give each of them about 1,950 items, past the 997 its
    directory page holds; the cores full take no more and the frame lays out."""
    tiles = 8160
    entries = np.zeros(tiles, np.int64)
    entries[:356] = 15358  # 260 x 15,358 + 6,800 = 4.0 M a tile
    heavy, band_items = card.heavy_capacity(held_of(tiles))
    library = build_middle(tmp_path, tiles, heavy, band_items)
    found, item_count, slots, directory = middle_of(
        library, tiles, heavy, band_items, entries
    )
    assert found == 0 and item_count == tiles and slots > 0
    own = directory[:, lists.K["kDirItems"]]
    assert own.max() == lists.K["kDirMaxItems"] and own.sum() == tiles
