"""The comparison of two rasterizer outputs: it must see and place differences."""

import numpy as np

from projects.raster.reference import bench, harness


def outputs():
    tri = np.full((64, 64), -1, dtype=np.int32)
    tri[10:30, 10:30] = 5
    return {
        "colour": np.zeros((64, 64, 3), dtype=np.float32),
        "depth": np.ones((64, 64), dtype=np.float32),
        "tri": tri,
        "tile_counts": np.ones((2, 2), dtype=np.int32),
    }


def test_identical_outputs_show_no_difference():
    stats = harness.compare(outputs(), outputs())
    assert stats["any_differs"] == 0 and stats["tile_counts_equal"]
    assert stats["colour_max_abs"] == 0 and not stats["mask"].any()


def test_differences_are_counted_by_array_and_placed():
    a, b = outputs(), outputs()
    b["tri"][10, 15] = 7  # on the border of the square
    b["depth"][20, 20] = 2.0  # inside it
    b["colour"][50, 50, 1] = 0.25  # far from any edge
    b["tile_counts"][0, 1] = 3
    stats = harness.compare(a, b)
    assert (stats["tri_differs"], stats["depth_differs"], stats["colour_differs"]) == (
        1,
        1,
        1,
    )
    assert stats["any_differs"] == 3 and stats["differs_on_edge"] == 1
    assert stats["differs_inside"] == 2
    assert stats["colour_max_abs"] == 0.25 and stats["depth_max_abs"] == 1.0
    assert not stats["tile_counts_equal"] and stats["tile_count_abs_diff"] == 2
    assert stats["mask"][10, 15] and stats["mask"][20, 20] and stats["mask"][50, 50]


def test_edge_mask_marks_the_border_of_regions():
    edge = harness.edge_mask(outputs()["tri"])
    assert edge[10, 10] and edge[9, 9] and edge[29, 29] and not edge[20, 20]
    assert not edge[50, 50]


def test_timing_summary_takes_the_median_over_all_repetitions():
    reps = [{k: v for k in bench.STAGES} for v in (1.0, 2.0, 9.0)]
    records = [
        {"reps": reps, "load_before": "1 2 3", "load_after": "4 5 6"},
        {"reps": reps, "load_before": "1 2 3", "load_after": "4 5 6"},
    ]
    s = bench.summarize(records)
    assert s["total_ms"] == {"median": 2.0, "min": 1.0, "max": 9.0, "n": 6}
    assert s["load"] == ["1 2 3 / 4 5 6"]
