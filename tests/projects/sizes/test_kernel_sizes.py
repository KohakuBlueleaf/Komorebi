"""Every project's program fits its session's kernel buffer with room to spare (no
card): each driver runs to its first load on a mock device, and the runtime's layout
of the five built kernels is compared with the buffer. KMR_SIZES=1 and TT_METAL_HOME
(a built tree); the native module and every program are built, minutes when cold."""

import pytest
from kernel_sizes_fixture import HEADROOM, build_sizes

pytestmark = pytest.mark.sizes

# name: arguments of the child (a driver script and its arguments, or --program)
CASES = {
    "raytrace": ["projects/raytrace/framegraph.py", "--tiles", "0", "300"],
    "attention": ["projects/attention/run.py", "--shape", "t3", "--calls", "2"],
    "dense": ["projects/dense/run.py", "--m", "256", "--k", "64", "--n", "128"],
    "dense_grid": [
        *("projects/dense/run.py", "--m", "512", "--k", "64", "--n", "384"),
        *("--form", "grid", "--grid", "2d:4,3,4,4,2,2,1"),
    ],
    "dense_split": [
        *("projects/dense/run.py", "--m", "160", "--k", "64", "--n", "96"),
        *("--split", "2,2,2,1"),
    ],
    "swiglu": ["projects/dense/swiglu.py", "--shape", "sw256"],
    "raster": ["--program", "projects.raster.card:program"],
    "raster_passes": ["projects/raster/card_passes.py", "--setup"],
    "raster_frame": [
        *("projects/raster/card_thread.py", "--scene", "s10k", "--steps", "4"),
    ],
    # programs written as classes: each built from its class, as made at its load
    "class_mapcollect": ["examples/01_mapcollect/as_class/mapcollect_class.py"],
    "class_merge_tree": ["examples/02_merge_tree/as_class/merge_class.py"],
    "class_broadcast": ["examples/03_broadcast/as_class/broadcast_class.py"],
    "class_movement": ["examples/11_movement/as_class/movement_class.py"],
    "class_overlays": ["examples/12_overlays/as_class/overlays_class.py"],
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_kernels_fit_the_buffer_with_room_to_spare(environment, name):
    row = build_sizes(environment, CASES[name])
    text = ", ".join(f"{thread} {size}" for thread, size in row["text"].items())
    assert (
        row["fits"] and row["total"] + HEADROOM <= row["limit"]
    ), f"{name}: {row['total']} of {row['limit']} B ({text})"
