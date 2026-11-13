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

FORMS = ("immediate", "data", "mask", "stack", "fast", "sync")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree("tiny")
    scene = scenes.load_scene(data / "tiny.npz")
    every = np.arange(scenes.WIDTH * scenes.HEIGHT)
    pixels = every[np.isin(packets.ray_order(every)[1], [528])][:32]
    origins, directions = raytrace.camera_rays(scene["camera"], pixels)
    lanes, directory, _ = packets.split(pixels, 1)
    pages = packets.packet_pages(lanes, origins, directions, raytrace.TMAX)
    report = {"passes": arguments.passes, "forms": {}}
    with card.open_session() as session:
        workers = card.workers(session)
        padded = np.zeros((workers, directory.shape[1]), dtype=np.uint32)
        padded[:1] = directory
        for form_index, form in enumerate(FORMS):
            assert packets.K[f"kProbe{form.capitalize()}"] == form_index
            _, profile, ran, _, _ = card.run_handler(
                session,
                packets.K["kHandlerProbe"],
                padded,
                packets.group(pages),
                1,
                packets.K["kPagePackets"] * packets.K["kHitAnyVectors"] * packets.VEC,
                tree=packets.tree_image(tree),
                args=[form_index, arguments.passes],
            )
            rows = card.profile_rows(profile, workers)
            visits = int(rows["inner"][0])
            cycles = int(rows["box"][0])
            report["forms"][form] = {
                "visits": visits,
                "cycles": cycles,
                "cycles_per_visit": cycles / max(visits, 1),
                "error": int(rows["error"][0]),
                "run": card.check_run(ran, workers),
            }
            card.log(f"{form}: {cycles / max(visits, 1):.1f} cycles per node visit")
    forms = report["forms"].values()
    report["ok"] = all(
        card.ran_cleanly(entry["run"], workers)
        and entry["error"] == 0
        and entry["visits"] > 0
        for entry in forms
    )
    print(
        json.dumps(
            {form: entry["cycles_per_visit"] for form, entry in report["forms"].items()}
        )
    )
    print(json.dumps({"ok": report["ok"]}), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=1)
    return report["ok"]


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
