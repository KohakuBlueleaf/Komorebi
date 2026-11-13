"""The arithmetic check's inputs on the CPU: edge cases, ties and the expected values
(the reference's own expressions)."""

import numpy as np
import pytest

from projects.raytrace import arithmetic, packets
from ref import bvh, raytrace, scenes

FLOAT32 = np.float32


@pytest.fixture(scope="module")
def rows():
    scene = scenes.make_preset("tiny")
    tree = bvh.build(scene)
    rng = np.random.default_rng(2)
    count = 2048
    origin = rng.uniform(-5, 5, (count, 3)).astype(FLOAT32)
    direction = rng.normal(size=(count, 3)).astype(FLOAT32)
    leaf = rng.integers(0, len(tree["tris"]), count)
    triangle = tree["tris"][leaf]
    lower, upper = bvh.node_boxes(tree["nodes"][tree["leaf_node"][leaf]])
    direction = (
        triangle[:, 0:3] + 0.3 * triangle[:, 3:6] + 0.3 * triangle[:, 6:9] - origin
    ).astype(FLOAT32)
    out = {
        "o": origin, "d": direction, "bmin": lower.copy(), "bmax": upper.copy(),
        "v0": triangle[:, 0:3].copy(), "e1": triangle[:, 3:6].copy(),
        "e2": triangle[:, 6:9].copy(),
        "tri": leaf.astype(np.int64), "best": np.full(count, raytrace.TMAX, FLOAT32),
    }  # fmt: skip
    edge = arithmetic.edge_inputs(rng, out, tree)
    arithmetic.bests(rng, out)
    return out, edge


def test_edge_cases_cover_their_kinds(rows):
    batch, edge = rows
    assert 0.15 < edge.mean() < 0.35
    assert (batch["d"] == 0).any() and np.signbit(batch["d"][batch["d"] == 0]).any()
    in_plane = (batch["bmax"] == batch["bmin"]) & (batch["o"] == batch["bmin"])
    assert in_plane.any(axis=1)[edge].sum() > 10
    hits, _, _, _ = arithmetic.triangle(batch)
    assert hits.any() and (~hits).any()


def test_bests_make_ties_and_neighbours(rows):
    batch, _ = rows
    hits, distance, _, _ = arithmetic.triangle(batch)
    tie = hits & (batch["best"] == distance)
    assert tie.sum() > 50
    assert (hits & (batch["best"] == np.nextafter(distance, FLOAT32(np.inf)))).any()
    assert set(np.unique(batch["cur"] - batch["tri"])) >= {-1, 0, 1}


def test_expected_matches_the_reference_trace_update(rows):
    batch, _ = rows
    want = arithmetic.expected(batch)
    hits, distance, _, _ = arithmetic.triangle(batch)
    closer = hits & (distance < batch["best"])
    assert np.array_equal(want["new_best"][closer], distance[closer])
    keep = ~hits
    assert np.array_equal(want["new_best"][keep], batch["best"][keep])


def test_pages_hold_the_rows(rows):
    batch, _ = rows
    pages = arithmetic.input_pages(batch)
    constants = packets.K
    assert pages.shape == (len(batch["o"]) // 32, constants["kArInVectors"] * 32)
    vectors = pages.reshape(-1, constants["kArInVectors"], 32)
    assert np.array_equal(
        vectors[:, constants["kArO"]].ravel().view(FLOAT32), batch["o"][:, 0]
    )


def test_math_inputs_stay_in_range_and_fill_pages():
    numerator, denominator, x = arithmetic.math_inputs(np.random.default_rng(3), 4096)
    quotient = numerator / denominator
    assert (
        (denominator > 0).all()
        and (x > 0).all()
        and (numerator == 0).any()
        and (numerator < 0).any()
    )
    normal = np.abs(quotient[numerator != 0]) >= np.finfo(FLOAT32).tiny
    assert normal.all() and np.isfinite(quotient).all()
    vectors = arithmetic.math_pages(numerator, denominator, x).reshape(
        -1, packets.K["kMaInVectors"], 32
    )
    assert np.array_equal(
        vectors[:, packets.K["kMaB"]].ravel().view(FLOAT32), denominator
    )
