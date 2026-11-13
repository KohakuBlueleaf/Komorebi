"""Result files of an implementation, external implementations, power records."""

import sys

import numpy as np
import pytest

from projects.raster.reference import bench, harness
from ref import power, results


def raster_arrays():
    tri = np.full((16, 16), -1, dtype=np.int32)
    tri[4:9, 4:9] = 3
    return {
        "colour": np.zeros((16, 16, 3), dtype=np.float32),
        "depth": (tri >= 0).astype(np.float32),
        "tile_counts": np.ones((2, 2), dtype=np.int32),
        "tri": tri,
    }


def test_results_round_trip_and_discovery(tmp_path):
    info = {"name": "card", "scene": "tiny", "seconds_per_frame": 0.5}
    results.save_result(tmp_path / "card", "tiny", "raster", raster_arrays(), info)
    arrays, again = results.load_result(tmp_path / "card", "tiny", "raster")
    assert again == info and np.array_equal(arrays["tri"], raster_arrays()["tri"])
    found = results.find_results([tmp_path, tmp_path / "missing"], "tiny", "raster")
    assert (
        list(found) == ["card"]
        and results.find_results([tmp_path], "s10k", "raster") == {}
    )


def test_a_missing_required_array_is_rejected(tmp_path):
    arrays = raster_arrays()
    del arrays["depth"]
    np.savez(tmp_path / "tiny.npz", **arrays)
    (tmp_path / "tiny.json").write_text('{"name": "x"}')
    with pytest.raises(ValueError, match="depth"):
        results.load_result(tmp_path, "tiny", "raster")


def test_an_external_result_without_tri_is_compared_and_shown_in_the_table(tmp_path):
    ref = raster_arrays()
    other = {name: array.copy() for name, array in ref.items() if name != "tri"}
    other["depth"][6, 6] = 0.5
    stats = harness.compare(ref, other)
    assert stats["tri_differs"] is None and stats["depth_differs"] == 1
    assert stats["any_differs"] == 1 and stats["mask"][6, 6]
    info = {"name": "card", "scene": "tiny", "seconds_per_frame": 0.01}
    info["timings_ms"] = {name: 1.0 for name in bench.STAGES}
    info["power"] = {"seconds_per_frame": 0.01, "gpu_joules_per_frame": 0.5}
    results.save_result(tmp_path / "card", "tiny", "raster", other, info)
    rows = bench.external_rows("tiny", [tmp_path])
    assert len(rows) == 1 and "card (external)" in rows[0] and "| 0.500 |" in rows[0]


def test_power_record_from_a_binary_that_marks_its_loop(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(
        "import json, sys, time\n"
        "print('LOOP_START', file=sys.stderr, flush=True)\n"
        "time.sleep(0.2)\n"
        "print('LOOP_END', file=sys.stderr, flush=True)\n"
        "print(json.dumps({'reps_n': 4, 'loop_ms': 200.0}))\n"
    )
    out = power.run_with_power([sys.executable, str(script)])
    record = out["power"]
    assert record["seconds_per_frame"] == pytest.approx(0.05)
    joules = record["cpu_package_joules_per_frame"]
    assert (joules is None) == (power.cpu_energy_uj() is None)
    assert joules is None or joules >= 0
    assert "load_before" in out and "gpu_mean_w" not in record
