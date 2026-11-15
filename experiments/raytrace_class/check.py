"""Off the card, after run_trace.py: the class's hit and profile pages against the hand
program's, word for word (the clock's words apart), and the class's hits against the
reference's trace of the same rays.
Example: python check.py --reference"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from projects.raytrace import card, packets, trace  # noqa: E402
from projects.raytrace.reference import harness  # noqa: E402
from reference import raytrace  # noqa: E402

K = packets.K
VECTORS = packets.hit_vectors(False)
# The profile's counters of cycles (the clock's) and of counts, by name (card.PROF).
COUNTS = ("packets", "pops", "empty", "inner", "leaves", "error")


def hit_pages(rows, count):
    """(packets, vectors, 32) hit pages from grouped rows."""
    return packets.ungroup(rows, VECTORS, count).reshape(count, VECTORS, packets.VEC)


def compare(hand, mine, count, workers):
    """Words that differ: the hit pages but their cycles vector, and the walk's
    counts in the profile (the class's page holds the pack CPU's words alone: the
    movers' counters are a line it does not have)."""
    a, b = hit_pages(hand["hits"], count), hit_pages(mine["hits"], count)
    cycles = K["kHitCycles"]
    keep = [vector for vector in range(VECTORS) if vector != cycles]
    differ = a[:, keep] != b[:, keep]
    result = {
        "hit_words": int(differ.size),
        "hit_words_differ": int(differ.sum()),
        "packets_differ": int(differ.any(axis=(1, 2)).sum()),
    }
    rows = {
        name: card.profile_rows(record["profile"], workers)
        for name, record in (("hand", hand), ("mine", mine))
    }
    result["profile_counts_differ"] = {
        name: int((rows["hand"][name] != rows["mine"][name]).sum()) for name in COUNTS
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", action="store_true", help="numpy's trace too")
    parser.add_argument("--workers", type=int, default=120)
    arguments = parser.parse_args()
    inputs = np.load(HERE / "build" / "inputs.npz")
    count = int(inputs["count"][0])
    records = {
        name: dict(np.load(HERE / "build" / f"{name}.npz"))
        for name in ("hand", "class")
        if (HERE / "build" / f"{name}.npz").exists()
    }
    for record in records.values():
        record["profile"] = np.asarray(record["profile"], np.uint64)
    mine = records["class"]
    report = {}
    if "hand" in records:
        report["compare"] = compare(records["hand"], mine, count, arguments.workers)
    for name, record in records.items():
        rows = card.profile_rows(record["profile"], arguments.workers)
        busy = rows["total"]
        report[name] = {
            "median_us": float(np.median(record["microseconds"])),
            "microseconds": [float(v) for v in record["microseconds"]],
            "busy_max": int(busy.max()),
            "busy_mean": float(busy.mean()),
            "sums": {key: int(values.sum()) for key, values in rows.items()},
        }
    if arguments.reference:
        tree = harness.ensure_tree("tiny")
        lanes = inputs["lanes"]
        origins, directions = inputs["origins"], inputs["directions"]
        rows = packets.ungroup(mine["hits"], VECTORS, count)
        hits = packets.decode_hits(rows, lanes, len(origins))
        hits.pop("attrs"), hits.pop("packet_cycles")
        leaf = trace.leaf_of(tree)
        valid = hits["tri"] >= 0
        hit = {
            "t": hits["t"],
            "leaf": np.where(valid, leaf[np.maximum(hits["tri"], 0)], -1),
            "u": hits["u"],
            "v": hits["v"],
            "nodes": hits["nodes"],
            "leaves": hits["leaves"],
        }
        reference = raytrace.trace(tree, origins, directions, raytrace.TMAX, False)
        report["reference"] = trace.agreement(reference, hit, tree, False)
    print(json.dumps(report, default=int), flush=True)


if __name__ == "__main__":
    main()
