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
from reference import native, scenes


class Recorder(trace.CardTracer):
    """The card tracer that keeps each pass's arguments for a later rerun."""

    def trace(self, name, origins, directions, tmax, any_hit, pixels, layout=None):
        self.kept = getattr(self, "kept", {})
        self.kept[name] = (origins, directions, tmax, any_hit, pixels)
        return super().trace(name, origins, directions, tmax, any_hit, pixels, layout)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--sort", choices=("none", "tile", "pass"), default="pass")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--passes", default="primary,shadow,bounce,shadow2")
    parser.add_argument(
        "--most", type=int, default=8000, help="commands per program at most"
    )
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()
    data = native.data_dir()
    tree = harness.ensure_tree(arguments.scene)
    scene = scenes.load_scene(data / f"{arguments.scene}.npz")
    pixels = np.arange(scenes.WIDTH * scenes.HEIGHT)
    index = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    report = {
        "scene": arguments.scene,
        "sort": arguments.sort,
        "seconds": arguments.seconds,
    }
    report["loops"] = []
    with card.open_session() as session:
        tracer = Recorder(
            session, tree, repetitions=3, balance=True, sort=arguments.sort
        )
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
            info["pass"]: np.median(info["trace_microseconds"])
            for info in tracer.passes[-4:]
        }
        for name in arguments.passes.split(","):
            microseconds = frame_microseconds[name]
            repetitions = min(
                arguments.most,
                max(1, math.ceil(arguments.seconds * 1e6 / microseconds)),
            )
            tracer.repetitions = repetitions
            tracer.power = PowerSampler(index)
            card.log(
                f"{name}: {repetitions} commands of {microseconds:.0f} us under the "
                "sampler"
            )
            tracer.trace(name, *tracer.kept[name])
            info, sampler, tracer.power = tracer.passes[-1], tracer.power, None
            power_phase = sampler.result.phases["program"].as_dict()
            power_phase["idle_mean_w"] = info["power"]["idle_mean_w"]
            power_phase["block_s"] = info["power"]["seconds"]
            power_phase["block_mean_w"] = info["power"]["mean_w"]
            loop = {"pass": name, "commands": repetitions, "rays": info["rays"]}
            loop["trace_microseconds_median"] = float(
                np.median(info["trace_microseconds"])
            )
            loop["card_s"] = float(np.sum(info["trace_microseconds"])) / 1e6
            loop["program_s"] = info["times"]["program_s"]
            loop.update(
                {
                    key: power_phase[key]
                    for key in ("seconds", "mean_w", "peak_w", "joules")
                }
            )
            loop.update(
                idle_w=power_phase["idle_mean_w"], too_short=power_phase["too_short"]
            )
            loop.update(
                block_s=power_phase["block_s"], block_mean_w=power_phase["block_mean_w"]
            )
            loop["walk_errors"] = info["walk_errors"]
            loop["run"] = info["run"]
            if power_phase["mean_w"] is not None:
                rays_per_second = info["rays"] / (
                    loop["trace_microseconds_median"] / 1e6
                )
                loop["mrays_per_s"] = rays_per_second / 1e6
                loop["nj_per_ray"] = power_phase["mean_w"] / rays_per_second * 1e9
            report["loops"].append(loop)
            print(
                json.dumps({key: value for key, value in loop.items() if key != "run"}),
                flush=True,
            )
    report["ok"] = report["frame_exact"] and all(
        card.ran_cleanly(loop["run"], tracer.workers) and loop["walk_errors"] == 0
        for loop in report["loops"]
    )
    print(json.dumps({"ok": report["ok"]}), flush=True)
    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=1, default=int)
    return report["ok"]


if __name__ == "__main__":
    started = time.time()
    ok = main()
    print(f"wall {time.time() - started:.1f}s", flush=True)
    sys.exit(0 if ok else 1)
