"""Cost of one node visit on one core: a packet of 32 primary rays against both
children of every inner node of the tiny tree, in each form of the probe.

    python -m projects.raytrace.probe [--passes 20] [--json FILE]
"""

import argparse
import json
import sys

import numpy as np

from projects.raytrace import card, packets
from projects.raytrace.reference import harness
from ref import native, raytrace, scenes

FORMS = ("imm", "data", "mask", "stack", "fast", "sync")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--passes", type=int, default=20)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree("tiny")
    scene = scenes.load_scene(data / "tiny.npz")
    every = np.arange(scenes.WIDTH * scenes.HEIGHT)
    pixels = every[np.isin(packets.ray_order(every)[1], [528])][:32]
    o, d = raytrace.camera_rays(scene["camera"], pixels)
    lanes, directory, _ = packets.split(pixels, 1)
    pages = packets.packet_pages(lanes, o, d, raytrace.TMAX)
    out = {"passes": args.passes, "forms": {}}
    with card.open_session() as session:
        workers = card.workers(session)
        full = np.zeros((workers, directory.shape[1]), dtype=np.uint32)
        full[:1] = directory
        for k, form in enumerate(FORMS):
            assert packets.K[f"kProbe{form.capitalize()}"] == k
            _, prof, res, _, _ = card.run_segment(
                session,
                packets.K["kSegProbe"],
                full,
                packets.group(pages),
                1,
                packets.K["kPagePackets"] * packets.K["kHitAnyVectors"] * packets.VEC,
                tree=packets.tree_image(tree),
                args=[k, args.passes],
            )
            rows = card.profile_rows(prof, workers)
            visits = int(rows["inner"][0])
            cycles = int(rows["box"][0])
            out["forms"][form] = {
                "visits": visits,
                "cycles": cycles,
                "cycles_per_visit": cycles / max(visits, 1),
                "error": int(rows["error"][0]),
                "run": card.check_run(res, workers),
            }
            card.log(f"{form}: {cycles / max(visits, 1):.1f} cycles per node visit")
    forms = out["forms"].values()
    out["ok"] = all(
        card.ran_cleanly(f["run"], workers) and f["error"] == 0 and f["visits"] > 0
        for f in forms
    )
    print(json.dumps({f: v["cycles_per_visit"] for f, v in out["forms"].items()}))
    print(json.dumps({"ok": out["ok"]}), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    return out["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
