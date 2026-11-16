"""The rasterizer on a card: tiles of the tiny scene equal to `raster.fill_tile`;
its frame driven by the work cluster's thread, with the reference's step, directory and
lists, COUNT and SETUP included (KMR_CARD=1)."""

import argparse

import pytest

from projects.raster import card, card_thread, lists, run

pytestmark = pytest.mark.card
CONSTANTS = lists.K


def fill(session, tiles, cores=0):
    """Fills the tiny scene's `tiles` on `cores` (0: all); the check of each tile."""
    scene, _ = run.load("tiny")
    prepared = run.prepare(scene, card.workers(session), tiles=tiles, cores=cores)
    n_out = len(prepared["tiles"]) * CONSTANTS["kPlanes"]
    lists_args = (
        CONSTANTS["kHandlerFill"],
        prepared["pages"],
        prepared["directory"],
        n_out,
        CONSTANTS["kPlanes"],
    )
    frame, _, status, _, _ = card.run_lists(session, *lists_args)
    assert status["ok"], status
    return prepared, run.check(
        "tiny", tiles, *(prepared[name] for name in run.ARGS), frame
    )


def test_tiles_of_the_tiny_scene_equal_fill_tile(session):
    # an empty tile, a corner, the scene's busiest tiles
    _, check = fill(session, [0, 367, 368, 400, 1023])
    assert run.check_ok(check), check


def test_the_thread_drives_a_frame_with_the_reference_lists(session):
    # the third frame's COUNT and SETUP run on ranges cut by the second one's costs
    args = argparse.Namespace(
        scene="s10k",
        cores=0,
        steps=4,
        list_slots=0,
        lists=64,
        image=False,
        balance=True,
        repeat=3,
    )
    out = card_thread.frame(args, session)
    assert out["thread"]["code"] == 0 and out["thread"]["where"] == 8, out
    assert out["lists"]["pages_differ"] == 0 and out["dir_equal"], out
    assert out["ok"], out


def test_the_thread_drives_a_frame_with_fewer_pass_cores_than_fill_cores(session):
    # PLACE still writes every fill core's directory page
    args = argparse.Namespace(
        scene="tiny",
        cores=7,
        steps=4,
        list_slots=0,
        lists=0,
        image=False,
        balance=True,
        repeat=0,
    )
    out = card_thread.frame(args, session)
    assert out["thread"]["code"] == 0 and out["thread"]["where"] == 8, out
    assert out["lists"]["pages_differ"] == 0 and out["dir_equal"], out
    assert out["ok"], out
