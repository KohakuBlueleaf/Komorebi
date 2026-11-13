"""Card power while tracing: a frame records each pass's rays, then each pass reruns
as one program lasting `--seconds`; power is taken over the program's own span.

    python -m projects.raytrace.powerloop --scene tiny --sort pass [--seconds 4]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from komorebi.power import PowerSampler
from projects.raytrace import card, trace
from projects.raytrace.reference import harness
from ref import native, scenes


class Recorder(trace.CardTracer):
    """The card tracer that keeps each pass's arguments for a later rerun."""

    def trace(self, name, o, d, tmax, any_hit, pixels, layout=None):
        self.kept = getattr(self, "kept", {})
        self.kept[name] = (o, d, tmax, any_hit, pixels)
        return super().trace(name, o, d, tmax, any_hit, pixels, layout)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--sort", choices=("none", "tile", "pass"), default="pass")
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--passes", default="primary,shadow,bounce,shadow2")
    p.add_argument(
        "--most", type=int, default=8000, help="commands per program at most"
    )
    p.add_argument("--json", default=None)
    args = p.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(args.scene)
    scene = scenes.load_scene(data / f"{args.scene}.npz")
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    index = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    report = {"scene": args.scene, "sort": args.sort, "seconds": args.seconds}
    report["loops"] = []
    with card.open_session() as session:
        tracer = Recorder(session, tree, reps=3, balance=True, sort=args.sort)
        for frame in range(2):  # the second frame is split by measured cost
            tracer.frame = frame
            trace.render_pixels(scene, tree, pixels, tracer)
        ok = all(
            not info["check"].get("tri_differs", info["check"].get("differ"))
            for info in tracer.passes
        )
        report["frame_exact"] = bool(ok)
        tracer.check = False
        frame_microseconds = {
            i["pass"]: np.median(i["trace_microseconds"]) for i in tracer.passes[-4:]
        }
        for name in args.passes.split(","):
            us = frame_microseconds[name]
            reps = min(args.most, max(1, math.ceil(args.seconds * 1e6 / us)))
            tracer.reps = reps
            tracer.power = PowerSampler(index)
            card.log(f"{name}: {reps} commands of {us:.0f} us under the sampler")
            tracer.trace(name, *tracer.kept[name])
            info, sampler, tracer.power = tracer.passes[-1], tracer.power, None
            pw = sampler.result.phases["program"].as_dict()
            pw["idle_mean_w"] = info["power"]["idle_mean_w"]
            pw["block_s"] = info["power"]["seconds"]
            pw["block_mean_w"] = info["power"]["mean_w"]
            loop = {"pass": name, "commands": reps, "rays": info["rays"]}
            loop["trace_microseconds_median"] = float(
                np.median(info["trace_microseconds"])
            )
            loop["card_s"] = float(np.sum(info["trace_microseconds"])) / 1e6
            loop["program_s"] = info["times"]["program_s"]
            loop.update({k: pw[k] for k in ("seconds", "mean_w", "peak_w", "joules")})
            loop.update(idle_w=pw["idle_mean_w"], too_short=pw["too_short"])
            loop.update(block_s=pw["block_s"], block_mean_w=pw["block_mean_w"])
            loop["walk_errors"] = info["walk_errors"]
            loop["run"] = info["run"]
            if pw["mean_w"] is not None:
                ray_s = info["rays"] / (loop["trace_microseconds_median"] / 1e6)
                loop["mrays_per_s"] = ray_s / 1e6
                loop["nj_per_ray"] = pw["mean_w"] / ray_s * 1e9
            report["loops"].append(loop)
            print(json.dumps({k: v for k, v in loop.items() if k != "run"}), flush=True)
    report["ok"] = report["frame_exact"] and all(
        card.ran_cleanly(loop["run"], tracer.workers) and loop["walk_errors"] == 0
        for loop in report["loops"]
    )
    print(json.dumps({"ok": report["ok"]}), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    t0 = time.time()
    ok = main()
    print(f"wall {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
