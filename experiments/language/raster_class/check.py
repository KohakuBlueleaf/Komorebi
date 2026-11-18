"""Off the card, after run_fill.py: the class's frame against the hand program's, word
for word, the profile pages' counts core by core, the times, and the class's frame
against the reference's image of the scene.
Example: python check.py --scene tiny"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.raster import run  # noqa: E402

# The profile's counts (the rest are cycles): the pack CPU's, the reader's, the
# writer's.
COUNTS = ("tiles", "tris", "row_count", "reader_pages", "writer_items")
CYCLES = ("init", "tri", "rows", "out", "total", "reader_total", "writer_total")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--workers", type=int, default=120)
    arguments = parser.parse_args()
    records = {
        name: dict(np.load(HERE / "build" / f"{name}.npz"))
        for name in ("hand", "class")
    }
    hand, mine = records["hand"], records["class"]
    rows = {
        name: run.profile_rows(record["profile"], arguments.workers)
        for name, record in records.items()
    }
    report = {
        "frame_words": int(hand["frame"].size),
        "frame_words_differ": int((hand["frame"] != mine["frame"]).sum()),
        "counts_differ": {
            name: int((rows["hand"][name] != rows["class"][name]).sum())
            for name in COUNTS
        },
        "cycles": {
            name: {key: int(rows[key][name].sum()) for key in rows} for name in CYCLES
        },
        "slowest_total": {key: int(rows[key]["total"].max()) for key in rows},
        "median_us": {
            name: float(np.median(record["microseconds"]))
            for name, record in records.items()
        },
    }
    scene, _ = run.load(arguments.scene)
    prepared = run.prepare(scene, arguments.workers)
    check = run.check(
        arguments.scene, None, *(prepared[key] for key in run.ARGS), mine["frame"]
    )
    report["reference"] = {key: value for key, value in check.items()}
    report["ok"] = report["frame_words_differ"] == 0 and run.check_ok(check)
    print(json.dumps(report, default=lambda value: value.tolist()), flush=True)
    if not report["ok"] and not hand["frame"].any() and not mine["frame"].any():
        raise SystemExit("check.py: both frames are zeros (reads were zeros)")
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
