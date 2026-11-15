"""Off the card, after run_passes.py: the class's counters, histograms and lists against
the hand program's, word for word, and both against the reference's: its counters and
histograms per core, and its records in the lists of the same middle pass.
Example: python check_passes.py --scene tiny"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(HERE)]

from run_passes import inputs  # noqa: E402

from projects.raster import lists, passes, run  # noqa: E402
from reference import raster  # noqa: E402


def wanted(scene, workers, given):
    """The reference's counters, the cut tiles' histograms and the lists' pages."""
    records = raster.setup(scene)
    triangles = len(scene["indices"])
    entries, rows = passes.counters(records, triangles, workers)
    starts, ends = passes.histograms(records, triangles, workers)
    cut = given["heavy"][0, 1 : 1 + given["cut"]]
    halves = np.stack([starts[:, cut], ends[:, cut]], axis=2).astype(np.uint16)
    _, offsets, members = raster.bin_triangles(records)
    packed = lists.pack_records(records)
    packed[:, lists.K["kRecIndex"]] = 0
    pages, _ = lists.build_lists(packed, offsets, members, given["middle"]["tiles_of"])
    return {
        "counters": np.concatenate([entries, rows], axis=1).astype(np.uint32),
        "histograms": halves.view(np.uint32).reshape(workers, -1),
        "lists": pages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--workers", type=int, default=120)
    arguments = parser.parse_args()
    records = {
        name: dict(np.load(HERE / "build" / f"passes_{name}.npz"))
        for name in ("hand", "class")
    }
    scene, _ = run.load(arguments.scene)
    given = inputs(scene, arguments.workers)
    want = wanted(scene, arguments.workers, given)
    report = {}
    for key in ("counters", "histograms", "lists"):
        hand, mine = (
            records[name][key].view(np.uint32).reshape(want[key].shape)
            for name in ("hand", "class")
        )
        report[key] = {
            "words": int(hand.size),
            "class_against_hand": int((hand != mine).sum()),
            "hand_against_reference": int((hand != want[key]).sum()),
        }
    report["microseconds"] = {
        name: dict(zip(("count", "hist", "setup"), record["microseconds"].tolist()))
        for name, record in records.items()
    }
    report["ok"] = all(
        part["class_against_hand"] == 0 and part["hand_against_reference"] == 0
        for part in (report[key] for key in ("counters", "histograms", "lists"))
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
