"""The middle pass between COUNT and SETUP in C (thread/middle.c) against the host's
(passes.py) on the same counts: items, directory, list headers and end slots, next
slots, item table; and the next slots SETUP makes from base words and SUM's and HSUM's
prefixes, and the directory and headers PLACE makes from rm_compact's words."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from projects.raster import lists, passes
from ref import raster, scenes

CONSTANTS = lists.K
THREAD = lists.HERE / "thread"
MAX_ITEMS, STRIDE = 2048, 2048
V, U = ctypes.c_void_p, ctypes.c_uint32


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("needs a C compiler (cc)")
    out = tmp_path_factory.mktemp("middle") / "libmiddle.so"
    command = [compiler, "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC"]
    subprocess.run([*command, str(THREAD / "middle.c"), "-o", str(out)], check=True)
    library = ctypes.CDLL(str(out))
    library.rm_heavy.argtypes = [V, U, U, V, V, V]
    library.rm_heavy.restype = U
    library.rm_middle.argtypes = [V, V, U, U, V, V, U, V, V, V, V, V, V, V, V, U]
    library.rm_middle.restype = U
    library.rm_compact.argtypes = [U, V, V, U, V, V]
    library.rm_compact.restype = U
    library.rm_base.argtypes = [V, V, U, V]
    library.rm_base.restype = None
    return library


def c_step(library, counts, rows, first, last, fill):
    """rm_heavy and rm_middle on the counts: (heavy, items, directory, marks, next,
    slots, item table)."""
    setup = len(counts)
    counters = np.ascontiguousarray(np.concatenate([counts, rows], axis=1), np.uint32)
    scratch = np.zeros(4 << 20, np.uint8)
    heavy, bands = np.zeros(256, np.uint32), np.zeros(256, np.uint32)
    heavy_count = library.rm_heavy(
        counters.ctypes.data, setup, fill, scratch.ctypes.data, heavy.ctypes.data,
        bands.ctypes.data,
    )  # fmt: skip
    heavy_tiles = heavy[:heavy_count].astype(int)
    hist = np.concatenate([first[:, heavy_tiles], last[:, heavy_tiles]], axis=2)
    hist = np.ascontiguousarray(hist, dtype=np.uint16)
    items = np.zeros((MAX_ITEMS, 4), np.uint32)
    tile_item = np.zeros(lists.TILES + 1, np.uint32)
    directory = np.zeros((fill, CONSTANTS["kDirBytes"] // 4), np.uint32)
    marks = np.zeros((MAX_ITEMS + 248, 6), np.uint32)
    next_slot = np.zeros((setup, STRIDE), np.uint32)
    item_total, mark_total = U(), U()
    slots = library.rm_middle(
        counters.ctypes.data, hist.ctypes.data, setup, fill, heavy.ctypes.data,
        bands.ctypes.data, heavy_count, scratch.ctypes.data, items.ctypes.data,
        ctypes.addressof(item_total), tile_item.ctypes.data, directory.ctypes.data,
        marks.ctypes.data, ctypes.addressof(mark_total), next_slot.ctypes.data, STRIDE,
    )  # fmt: skip
    item_count = item_total.value
    item_words = np.zeros((MAX_ITEMS, 2), np.uint32)
    core_words = np.zeros((fill, 4), np.uint32)
    compact = library.rm_compact(
        fill, scratch.ctypes.data, items.ctypes.data, item_count,
        item_words.ctypes.data, core_words.ctypes.data,
    )  # fmt: skip
    base = np.zeros(MAX_ITEMS, np.uint32)
    library.rm_base(
        scratch.ctypes.data, items.ctypes.data, item_count, base.ctypes.data
    )
    return {
        "base": base[:item_count],
        "compact": (compact, item_words[:item_count], core_words),
        "hist": hist,
        "heavy": list(zip(heavy_tiles.tolist(), bands[:heavy_count].tolist())),
        "items": items[:item_count],
        "directory": directory,
        "marks": marks[: mark_total.value],
        "next": next_slot[:, :item_count],
        "slots": slots,
        "tile_item": tile_item,
    }


def from_prefixes(result, counts):
    """Each setup core's next slots as SETUP's reader makes them with kSetupPrefix:
    base word, plus the entries before the core in a whole tile, plus for a heavy
    tile's rows the starts before row1 less the ends before row0 of the cores before."""
    bits = 23
    slot, band = result["base"] & ((1 << bits) - 1), result["base"] >> bits
    before = np.cumsum(counts, axis=0) - counts  # SUM's
    hist = result["hist"].astype(np.int64)
    hbefore = np.cumsum(hist, axis=0) - hist  # HSUM's, before the groups' sums
    starts = np.cumsum(hbefore[:, :, : passes.GROUPS], axis=2)
    ends = np.cumsum(hbefore[:, :, passes.GROUPS :], axis=2)
    out = np.zeros((len(counts), len(slot)), np.int64)
    for i, (tile_id, first_row, last_row) in enumerate(result["items"][:, :3].tolist()):
        if band[i] == 0:
            out[:, i] = slot[i] + before[:, tile_id]
            continue
        last_group, first_group = last_row // 2, first_row // 2
        add = starts[:, band[i] - 1, last_group - 1] if last_group else 0
        out[:, i] = (
            slot[i]
            + add
            - (ends[:, band[i] - 1, first_group - 1] if first_group else 0)
        )
    return out


def placed(item_words, core_words, items, slots):
    """The directory pages and list headers PLACE makes from rm_compact's words."""
    directory = np.zeros((len(core_words), CONSTANTS["kDirBytes"] // 4), np.uint32)
    pages = np.zeros((slots, lists.SLOT), np.uint32)
    for result, (first, n_pages, item_count, end) in enumerate(core_words.tolist()):
        directory[result, : CONSTANTS["kDirIds"]] = first, n_pages, item_count
        pages[end, 0] = CONSTANTS["kMarkEnd"]
    cursor = np.zeros(len(core_words), int)
    for (result, page), (tile_id, first_row, last_row, total) in zip(
        item_words.tolist(), items.tolist()
    ):
        core, entries = result & 0xFF, result >> 8
        assert entries == total
        tile = tile_id
        if (first_row, last_row) != (0, 32):
            tile |= (
                first_row << CONSTANTS["kItemRow0Shift"]
                | last_row << CONSTANTS["kItemRow1Shift"]
            )
        directory[core, CONSTANTS["kDirIds"] + cursor[core]] = tile | (
            0 if total else CONSTANTS["kItemEmpty"]
        )
        cursor[core] += 1
        if total:
            header_columns = [
                0,
                CONSTANTS["kHdrTile"],
                CONSTANTS["kHdrCount"],
                CONSTANTS["kHdrRow0"],
                CONSTANTS["kHdrRow1"],
            ]
            pages[page, header_columns] = (
                CONSTANTS["kMarkTile"],
                tile_id,
                total,
                first_row,
                last_row,
            )
    return directory, pages


def same(result, counts, rows, first, last, fill):
    """Asserts that the C step equals passes' on the counts."""
    heavy = passes.heavy_tiles(counts, rows, fill)
    assert result["heavy"] == heavy
    heavy_tiles = [tile_id for tile_id, _ in heavy]
    bands = passes.items_of(
        counts, rows, heavy, first[:, heavy_tiles], last[:, heavy_tiles]
    )
    mid = passes.middle(counts, rows, fill, bands=bands)
    assert [tuple(i[:3]) for i in result["items"].tolist()] == [
        tuple(map(int, i)) for i in mid["items"]
    ]
    assert np.array_equal(result["items"][:, 3], bands[1].sum(0))
    assert np.array_equal(result["directory"], mid["directory"])
    assert result["slots"] == mid["slots"]
    assert np.array_equal(result["next"], mid["next"])
    assert np.array_equal(from_prefixes(result, counts), mid["next"])
    pages = np.zeros((mid["slots"], lists.SLOT), np.uint32)
    for page, mark, tile_id, item_count, first_row, last_row in result[
        "marks"
    ].tolist():
        pages[page, 0] = mark
        if mark == CONSTANTS["kMarkTile"]:
            header_columns = [
                CONSTANTS["kHdrTile"],
                CONSTANTS["kHdrCount"],
                CONSTANTS["kHdrRow0"],
                CONSTANTS["kHdrRow1"],
            ]
            pages[page, header_columns] = tile_id, item_count, first_row, last_row
    assert np.array_equal(pages.reshape(mid["pages"].shape), mid["pages"])
    assert result["compact"][0] == mid["slots"]
    directory, pages_view = placed(
        result["compact"][1], result["compact"][2], result["items"], len(pages)
    )
    assert np.array_equal(directory, mid["directory"])
    assert np.array_equal(pages_view.reshape(mid["pages"].shape), mid["pages"])
    # the thread's items page, from tile_item and the items' rows
    tile_item = result["tile_item"]
    table = passes.item_table(mid["items"])
    first_n = tile_item[:-1] | (np.diff(tile_item) << 16)
    assert np.array_equal(first_n, table[: lists.TILES])
    return heavy


def test_the_c_step_equals_the_host_step_on_a_scene(lib):
    scene = scenes.make_preset("tiny")
    recs = raster.setup(scene)
    item_count = len(scene["indices"])
    counts, rows = passes.counters(recs, item_count, 7)
    first, last = passes.histograms(recs, item_count, 7)
    for fill in (5, 120):
        same(
            c_step(lib, counts, rows, first, last, fill),
            counts,
            rows,
            first,
            last,
            fill,
        )


def test_the_c_step_cuts_hot_tiles_into_the_host_steps_bands(lib):
    # entries of 7 cores, most of them in three tiles: those tiles are cut
    rng = np.random.default_rng(5)
    cores, item_count = 7, 6000
    core = rng.integers(0, cores, item_count)
    tile = np.where(
        rng.random(item_count) < 0.7,
        rng.choice([5, 400, 401], item_count),
        rng.integers(0, 1024, item_count),
    )
    first_row = rng.integers(0, 32, item_count)
    last_row = np.minimum(31, first_row + rng.integers(0, 12, item_count))
    base = (core * lists.TILES + tile) * passes.GROUPS
    size = cores * lists.TILES * passes.GROUPS
    shape = (cores, lists.TILES, passes.GROUPS)
    first = np.bincount(base + first_row // 2, minlength=size).reshape(shape)
    last = np.bincount(base + last_row // 2, minlength=size).reshape(shape)
    counts = np.bincount(core * lists.TILES + tile, minlength=cores * lists.TILES)
    counts = counts.reshape(cores, lists.TILES)
    rows = np.bincount(
        core * lists.TILES + tile,
        weights=last_row - first_row + 1,
        minlength=cores * lists.TILES,
    )
    rows = rows.reshape(cores, lists.TILES).astype(np.int64)
    result = c_step(lib, counts, rows, first, last, 40)
    heavy = same(result, counts, rows, first, last, 40)
    assert len(heavy) >= 3 and len(result["items"]) > lists.TILES
