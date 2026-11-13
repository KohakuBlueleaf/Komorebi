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

K = lists.K
THREAD = lists.HERE / "thread"
MAX_ITEMS, STRIDE = 2048, 2048
V, U = ctypes.c_void_p, ctypes.c_uint32


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("needs a C compiler (cc)")
    out = tmp_path_factory.mktemp("middle") / "libmiddle.so"
    cmd = [cc, "-std=c11", "-O2", "-Wall", "-Werror", "-shared", "-fPIC"]
    subprocess.run([*cmd, str(THREAD / "middle.c"), "-o", str(out)], check=True)
    so = ctypes.CDLL(str(out))
    so.rm_heavy.argtypes = [V, U, U, V, V, V]
    so.rm_heavy.restype = U
    so.rm_middle.argtypes = [V, V, U, U, V, V, U, V, V, V, V, V, V, V, V, U]
    so.rm_middle.restype = U
    so.rm_compact.argtypes = [U, V, V, U, V, V]
    so.rm_compact.restype = U
    so.rm_base.argtypes = [V, V, U, V]
    so.rm_base.restype = None
    return so


def c_step(so, cnt, rows, first, last, fill):
    """rm_heavy and rm_middle on the counts: (heavy, items, directory, marks, next,
    slots, item table)."""
    setup = len(cnt)
    counters = np.ascontiguousarray(np.concatenate([cnt, rows], axis=1), np.uint32)
    scratch = np.zeros(4 << 20, np.uint8)
    heavy, bands = np.zeros(256, np.uint32), np.zeros(256, np.uint32)
    nh = so.rm_heavy(
        counters.ctypes.data, setup, fill, scratch.ctypes.data, heavy.ctypes.data,
        bands.ctypes.data,
    )  # fmt: skip
    ht = heavy[:nh].astype(int)
    hist = np.concatenate([first[:, ht], last[:, ht]], axis=2)
    hist = np.ascontiguousarray(hist, dtype=np.uint16)
    items = np.zeros((MAX_ITEMS, 4), np.uint32)
    tile_item = np.zeros(lists.TILES + 1, np.uint32)
    d = np.zeros((fill, K["kDirBytes"] // 4), np.uint32)
    marks = np.zeros((MAX_ITEMS + 248, 6), np.uint32)
    nxt = np.zeros((setup, STRIDE), np.uint32)
    ni, nm = U(), U()
    slots = so.rm_middle(
        counters.ctypes.data, hist.ctypes.data, setup, fill, heavy.ctypes.data,
        bands.ctypes.data, nh, scratch.ctypes.data, items.ctypes.data,
        ctypes.addressof(ni), tile_item.ctypes.data, d.ctypes.data, marks.ctypes.data,
        ctypes.addressof(nm), nxt.ctypes.data, STRIDE,
    )  # fmt: skip
    n = ni.value
    item_words = np.zeros((MAX_ITEMS, 2), np.uint32)
    core_words = np.zeros((fill, 4), np.uint32)
    compact = so.rm_compact(
        fill, scratch.ctypes.data, items.ctypes.data, n, item_words.ctypes.data,
        core_words.ctypes.data,
    )  # fmt: skip
    base = np.zeros(MAX_ITEMS, np.uint32)
    so.rm_base(scratch.ctypes.data, items.ctypes.data, n, base.ctypes.data)
    return {
        "base": base[:n],
        "compact": (compact, item_words[:n], core_words),
        "hist": hist,
        "heavy": list(zip(ht.tolist(), bands[:nh].tolist())),
        "items": items[:n],
        "directory": d,
        "marks": marks[: nm.value],
        "next": nxt[:, :n],
        "slots": slots,
        "tile_item": tile_item,
    }


def from_prefixes(c, cnt):
    """Each setup core's next slots as SETUP's reader makes them with kSetupPrefix:
    base word, plus the entries before the core in a whole tile, plus for a heavy
    tile's rows the starts before row1 less the ends before row0 of the cores before."""
    bits = 23
    slot, h = c["base"] & ((1 << bits) - 1), c["base"] >> bits
    before = np.cumsum(cnt, axis=0) - cnt  # SUM's
    hist = c["hist"].astype(np.int64)
    hbefore = np.cumsum(hist, axis=0) - hist  # HSUM's, before the groups' sums
    starts = np.cumsum(hbefore[:, :, : passes.GROUPS], axis=2)
    ends = np.cumsum(hbefore[:, :, passes.GROUPS :], axis=2)
    out = np.zeros((len(cnt), len(slot)), np.int64)
    for i, (t, r0, r1) in enumerate(c["items"][:, :3].tolist()):
        if h[i] == 0:
            out[:, i] = slot[i] + before[:, t]
            continue
        g1, g0 = r1 // 2, r0 // 2
        add = starts[:, h[i] - 1, g1 - 1] if g1 else 0
        out[:, i] = slot[i] + add - (ends[:, h[i] - 1, g0 - 1] if g0 else 0)
    return out


def placed(item_words, core_words, items, slots):
    """The directory pages and list headers PLACE makes from rm_compact's words."""
    d = np.zeros((len(core_words), K["kDirBytes"] // 4), np.uint32)
    pages = np.zeros((slots, lists.SLOT), np.uint32)
    for c, (first, n_pages, n, end) in enumerate(core_words.tolist()):
        d[c, : K["kDirIds"]] = first, n_pages, n
        pages[end, 0] = K["kMarkEnd"]
    k = np.zeros(len(core_words), int)
    for (c, at), (t, r0, r1, total) in zip(item_words.tolist(), items.tolist()):
        core, entries = c & 0xFF, c >> 8
        assert entries == total
        tile = t
        if (r0, r1) != (0, 32):
            tile |= r0 << K["kItemRow0Shift"] | r1 << K["kItemRow1Shift"]
        d[core, K["kDirIds"] + k[core]] = tile | (0 if total else K["kItemEmpty"])
        k[core] += 1
        if total:
            hdr = [0, K["kHdrTile"], K["kHdrCount"], K["kHdrRow0"], K["kHdrRow1"]]
            pages[at, hdr] = K["kMarkTile"], t, total, r0, r1
    return d, pages


def same(c, cnt, rows, first, last, fill):
    """Asserts that the C step equals passes' on the counts."""
    heavy = passes.heavy_tiles(cnt, rows, fill)
    assert c["heavy"] == heavy
    ht = [t for t, _ in heavy]
    bands = passes.items_of(cnt, rows, heavy, first[:, ht], last[:, ht])
    mid = passes.middle(cnt, rows, fill, bands=bands)
    assert [tuple(i[:3]) for i in c["items"].tolist()] == [
        tuple(map(int, i)) for i in mid["items"]
    ]
    assert np.array_equal(c["items"][:, 3], bands[1].sum(0))
    assert np.array_equal(c["directory"], mid["directory"])
    assert c["slots"] == mid["slots"]
    assert np.array_equal(c["next"], mid["next"])
    assert np.array_equal(from_prefixes(c, cnt), mid["next"])
    pages = np.zeros((mid["slots"], lists.SLOT), np.uint32)
    for at, mark, t, n, r0, r1 in c["marks"].tolist():
        pages[at, 0] = mark
        if mark == K["kMarkTile"]:
            hdr = [K["kHdrTile"], K["kHdrCount"], K["kHdrRow0"], K["kHdrRow1"]]
            pages[at, hdr] = t, n, r0, r1
    assert np.array_equal(pages.reshape(mid["pages"].shape), mid["pages"])
    assert c["compact"][0] == mid["slots"]
    d, p = placed(c["compact"][1], c["compact"][2], c["items"], len(pages))
    assert np.array_equal(d, mid["directory"])
    assert np.array_equal(p.reshape(mid["pages"].shape), mid["pages"])
    # the thread's items page, from tile_item and the items' rows
    tile_item = c["tile_item"]
    table = passes.item_table(mid["items"])
    first_n = tile_item[:-1] | (np.diff(tile_item) << 16)
    assert np.array_equal(first_n, table[: lists.TILES])
    return heavy


def test_the_c_step_equals_the_host_step_on_a_scene(lib):
    scene = scenes.make_preset("tiny")
    recs = raster.setup(scene)
    n = len(scene["indices"])
    cnt, rows = passes.counters(recs, n, 7)
    first, last = passes.histograms(recs, n, 7)
    for fill in (5, 120):
        same(c_step(lib, cnt, rows, first, last, fill), cnt, rows, first, last, fill)


def test_the_c_step_cuts_hot_tiles_into_the_host_steps_bands(lib):
    # entries of 7 cores, most of them in three tiles: those tiles are cut
    rng = np.random.default_rng(5)
    cores, n = 7, 6000
    core = rng.integers(0, cores, n)
    tile = np.where(
        rng.random(n) < 0.7, rng.choice([5, 400, 401], n), rng.integers(0, 1024, n)
    )
    r0 = rng.integers(0, 32, n)
    r1 = np.minimum(31, r0 + rng.integers(0, 12, n))
    at = (core * lists.TILES + tile) * passes.GROUPS
    size = cores * lists.TILES * passes.GROUPS
    shape = (cores, lists.TILES, passes.GROUPS)
    first = np.bincount(at + r0 // 2, minlength=size).reshape(shape)
    last = np.bincount(at + r1 // 2, minlength=size).reshape(shape)
    cnt = np.bincount(core * lists.TILES + tile, minlength=cores * lists.TILES)
    cnt = cnt.reshape(cores, lists.TILES)
    rows = np.bincount(
        core * lists.TILES + tile, weights=r1 - r0 + 1, minlength=cores * lists.TILES
    )
    rows = rows.reshape(cores, lists.TILES).astype(np.int64)
    c = c_step(lib, cnt, rows, first, last, 40)
    heavy = same(c, cnt, rows, first, last, 40)
    assert len(heavy) >= 3 and len(c["items"]) > lists.TILES
