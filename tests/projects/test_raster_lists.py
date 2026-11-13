"""Host side of the host-binned rasterizer: records, tile assignment, lists, frames."""

import numpy as np
import pytest

from projects.raster import lists
from ref import raster, scenes

CONSTANTS = lists.K
FLOAT32 = np.float32


@pytest.fixture(scope="module")
def tiny():
    recs = raster.setup(scenes.make_preset("tiny"))
    counts, offsets, members = raster.bin_triangles(recs)
    return recs, counts, offsets, members, lists.pack_records(recs)


def test_bands_of_a_split_tile_are_even_rows_covering_the_tile(tiny):
    recs, counts, offsets, members, cards = tiny
    cost = lists.tile_cost(cards, offsets, members)
    t = int(np.argmax(counts))
    items, item_cost = lists.split_tiles(cards, offsets, members, [t], cost, 9000)
    bands = [(a, b) for u, a, b in items if u == t]
    assert len(bands) > 1 and bands[0][0] == 0 and bands[-1][1] == 32
    assert all(a % 2 == 0 and b % 2 == 0 and a < b for a, b in bands)
    assert all(bands[i][1] == bands[i + 1][0] for i in range(len(bands) - 1))
    assert len(item_cost) == len(items)


def test_band_items_carry_their_rows_in_header_and_id(tiny):
    recs, counts, offsets, members, cards = tiny
    t = int(np.argmax(counts))
    tiles_of = [[(t, 0, 10), (t, 10, 32)], [0]]
    pages, directory = lists.build_lists(cards, offsets, members, tiles_of)
    ids = directory[0, CONSTANTS["kDirIds"] : CONSTANTS["kDirIds"] + 2]
    for (_, a, b), i in zip(tiles_of[0], ids):
        assert i & CONSTANTS["kItemSlot"] == t
        assert i >> CONSTANTS["kItemRow0Shift"] & CONSTANTS["kItemRowMask"] == a
        assert i >> CONSTANTS["kItemRow1Shift"] & CONSTANTS["kItemRowMask"] == b
    slots = pages[: directory[0, CONSTANTS["kDirPages"]]].reshape(-1, lists.SLOT)
    assert (
        slots[0, CONSTANTS["kHdrRow0"]] == 0 and slots[0, CONSTANTS["kHdrRow1"]] == 10
    )
    n0 = slots[0, CONSTANTS["kHdrCount"]]
    assert (
        slots[1 + n0, CONSTANTS["kHdrRow0"]] == 10
        and slots[1 + n0, CONSTANTS["kHdrRow1"]] == 32
    )
    whole = directory[1, CONSTANTS["kDirIds"]]
    assert whole & ~np.uint32(CONSTANTS["kItemEmpty"]) == 0


def test_records_carry_thresholds_and_box(tiny):
    recs, *_, cards = tiny
    tl = recs["top_left"]
    t = cards[:, CONSTANTS["kRecT"] : CONSTANTS["kRecT"] + 3]
    assert np.array_equal(t == lists.NEG_ZERO_NEIGHBOUR, tl)
    assert np.array_equal(t[~tl], np.zeros((~tl).sum(), np.uint32))
    box = recs["box"]
    half = (
        cards[:, [CONSTANTS["kRecX0"], CONSTANTS["kRecX1"]]]
        .astype(np.uint16)
        .view(np.float16)
    )
    assert np.array_equal(half[:, 0].astype(FLOAT32), box[:, 0].astype(FLOAT32))
    assert np.array_equal(half[:, 1].astype(FLOAT32), (box[:, 1] + 1).astype(FLOAT32))
    assert np.array_equal(cards[:, CONSTANTS["kRecY1"]], box[:, 3])
    assert np.array_equal(
        cards[:, CONSTANTS["kRecTri"]].view(FLOAT32), recs["tri"].astype(FLOAT32)
    )


@pytest.mark.parametrize("mode", ["round-robin", "balanced"])
def test_every_tile_goes_to_exactly_one_core(tiny, mode):
    counts = tiny[1]
    parts = lists.assign(counts, 7, mode)
    flat = sorted(t for p in parts for t in p)
    assert flat == list(range(lists.TILES)) and len(parts) == 7


def test_lists_hold_each_tile_header_then_its_records(tiny):
    recs, counts, offsets, members, cards = tiny
    tiles_of = lists.assign(counts, 5, "round-robin")
    pages, directory = lists.build_lists(cards, offsets, members, tiles_of)
    for core, tiles in enumerate(tiles_of):
        first, n = (
            directory[core, CONSTANTS["kDirFirst"]],
            directory[core, CONSTANTS["kDirPages"]],
        )
        assert directory[core, CONSTANTS["kDirItems"]] == len(tiles)
        ids = directory[core, CONSTANTS["kDirIds"] : CONSTANTS["kDirIds"] + len(tiles)]
        assert list(ids & ~np.uint32(CONSTANTS["kItemEmpty"])) == tiles
        slots = pages[first : first + n].reshape(-1, lists.SLOT)
        at = 0
        for t, i in zip(tiles, ids):
            empty = counts[t] == 0
            assert bool(i & CONSTANTS["kItemEmpty"]) == empty
            if empty:
                continue  # no header: the writer stores the empty planes
            assert (
                slots[at, 0] == CONSTANTS["kMarkTile"]
                and slots[at, CONSTANTS["kHdrTile"]] == t
            )
            m = members[offsets[t] : offsets[t + 1]]
            assert slots[at, CONSTANTS["kHdrCount"]] == len(m)
            assert np.array_equal(slots[at + 1 : at + 1 + len(m)], cards[m])
            at += 1 + len(m)
        assert slots[at, 0] == CONSTANTS["kMarkEnd"]
        assert len(slots) - at <= lists.PAGE_SLOTS
    assert directory[-1, CONSTANTS["kDirFirst"]] + directory[
        -1, CONSTANTS["kDirPages"]
    ] == len(pages)


def test_a_frame_decodes_to_the_images_it_was_made_from():
    rng = np.random.default_rng(3)
    img = {
        "colour": rng.random((raster.HEIGHT, raster.WIDTH, 3)).astype(FLOAT32),
        "depth": rng.random((raster.HEIGHT, raster.WIDTH)).astype(FLOAT32),
        "tri": rng.integers(-1, 1 << 20, (raster.HEIGHT, raster.WIDTH)).astype(
            np.int32
        ),
    }
    back = lists.decode_frame(lists.frame_of(img))
    for key in img:
        assert np.array_equal(back[key], img[key])


def test_window_order_is_a_permutation_of_the_tile():
    assert sorted(lists.TILE_ORDER.ravel()) == list(range(1024))
    # Rows 0 and 1 share the first 64 words, even and odd columns.
    assert lists.TILE_ORDER[0, :8].tolist() == list(range(0, 16, 2))
    assert lists.TILE_ORDER[1, :8].tolist() == list(range(1, 16, 2))
