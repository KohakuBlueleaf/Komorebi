"""The two passes' host side: mesh blocks, counters and the step between COUNT and
SETUP, whose list positions must give the lists of `lists.build_lists`."""

import numpy as np
import pytest

from projects.raster import lists, passes
from ref import raster, scenes

K = lists.K


@pytest.fixture(scope="module")
def tiny():
    scene = scenes.make_preset("tiny")
    recs = raster.setup(scene)
    return scene, recs, raster.bin_triangles(recs)


def test_a_block_holds_32_triangles_in_window_order(tiny):
    scene = tiny[0]
    blocks = passes.mesh_blocks(scene)
    n = len(scene["indices"])
    assert blocks.shape == (-(-n // 32), passes.BLOCK_WORDS)
    words = passes.window_words(passes.BLOCK_SLOTS)
    vec = blocks[:, words].view(np.float32)  # (block, slot, lane)
    tri = 37
    corners = scene["positions"][scene["indices"][tri]]
    colours = scene["colors"][scene["indices"][tri]]
    got = vec[tri // 32, :, tri % 32]
    assert np.array_equal(got[:9], corners.reshape(-1))
    assert np.array_equal(got[9:], colours.reshape(-1))
    assert not vec[-1, :, n % 32 :].any()  # padding: corners all 0


def test_partitions_are_contiguous_and_cover_every_block():
    edge = passes.shares(1000, 120)
    assert edge[0] == 0 and edge[-1] == 1000 and (np.diff(edge) >= 0).all()
    assert np.diff(edge).max() - np.diff(edge).min() <= 1


def test_counters_sum_to_the_tile_lists_and_give_their_cost(tiny):
    scene, recs, (counts, offsets, members) = tiny
    cnt, rows = passes.counters(recs, len(scene["indices"]), 7)
    assert np.array_equal(cnt.sum(axis=0), counts)
    cards = lists.pack_records(recs)
    want = lists.tile_cost(cards, offsets, members)
    assert np.array_equal(passes.cost_of(cnt, rows), want)


def test_setup_at_the_middle_steps_slots_writes_the_lists_of_build_lists(tiny):
    scene, recs, (counts, offsets, members) = tiny
    n, cores = len(scene["indices"]), 7
    cnt, rows = passes.counters(recs, n, cores)
    mid = passes.middle(cnt, rows, 5)
    cards = lists.pack_records(recs)
    cards[:, K["kRecIndex"]] = 0
    want, directory = lists.build_lists(cards, offsets, members, mid["tiles_of"])
    assert np.array_equal(directory, mid["directory"])
    # SETUP: each core's records in triangle order, each to the next slot of its tiles
    slots = mid["pages"].reshape(-1, lists.SLOT).copy()
    nxt = mid["next"].astype(np.int64)
    edge = passes.shares(-(-n // 32), cores) * 32
    core = np.searchsorted(edge, recs["tri"], side="right") - 1
    box = recs["box"] // raster.TILE
    for i in range(len(box)):
        for ty in range(box[i, 2], box[i, 3] + 1):
            for tx in range(box[i, 0], box[i, 1] + 1):
                t = ty * raster.TILES_X + tx
                assert nxt[core[i], t] < mid["slots"]
                slots[nxt[core[i], t]] = cards[i]
                nxt[core[i], t] += 1
    assert np.array_equal(slots.reshape(want.shape), want)
