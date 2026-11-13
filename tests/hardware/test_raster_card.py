"""The rasterizer on a card through the host package, one session: a few tiles of the
tiny scene, and one tile split into bands over two cores, each equal to
`raster.fill_tile`; COUNT and SETUP of the tiny scene, whose counters and lists equal
the reference's (KMR_CARD=1)."""

import argparse

import pytest

import komorebi as km
from projects.raster import card, card_passes, lists, run

pytestmark = pytest.mark.card
CARD_IN_PROCESS = True  # opens the card in this process
K = lists.K


@pytest.fixture(scope="module")
def session():
    with km.Session(log=card.log) as s:
        yield s


def fill(session, tiles, cores=0):
    """Fills the tiny scene's `tiles` on `cores` (0: all); the check of each tile."""
    scene, _ = run.load("tiny")
    h = run.prepare(scene, card.workers(session), tiles=tiles, cores=cores)
    n_out = len(h["tiles"]) * K["kPlanes"]
    lists_args = (K["kSegFill"], h["pages"], h["directory"], n_out, K["kPlanes"])
    frame, _, status, _, _ = card.run_lists(session, *lists_args)
    assert status["ok"], status
    return h, run.check("tiny", tiles, *(h[k] for k in run.ARGS), frame)


def test_tiles_of_the_tiny_scene_equal_fill_tile(session):
    # an empty tile, a corner, the scene's busiest tiles
    _, check = fill(session, [0, 367, 368, 400, 1023])
    assert run.check_ok(check), check


def test_a_tile_split_into_bands_over_two_cores_equals_fill_tile(session):
    h, check = fill(session, [367], cores=2)
    assert len(h["tiles_of"][0]) + len(h["tiles_of"][1]) > 1
    assert run.check_ok(check), check


def test_count_and_setup_of_the_tiny_scene_write_the_reference_lists(session):
    args = argparse.Namespace(scene="tiny", cores=0, setup=True, lists=0)
    out = card_passes.frame(args, session)
    assert out["counters_equal"] and out["lists"]["pages_differ"] == 0, out
    assert out["ok"], out
