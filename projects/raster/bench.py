"""Frame time and power of the rasterizer on a card: host steps, then FILL commands back
to back in one run under the power and clock samplers, power over the stretches the
commands ran (see the README's "Time"). `--out` writes the results layout.
`python -m projects.raster.bench --scene s10k [--seconds 5] [--json F] [--out DIR]`"""

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import power
from projects.raster import card, lists, run

K = lists.K


def host_times(scene, workers, mode, repetitions, split):
    """Median seconds of each host step over `repetitions` preparations, and the last
    one."""
    steps, prepared = [], None
    for _ in range(repetitions):
        prepared = run.prepare(scene, workers, mode, split=split)
        steps.append(prepared["host"])
    return {
        step: float(np.median([sample[step] for sample in steps])) for step in steps[0]
    }, prepared


def card_index():
    """The card this process runs on (the first of TT_VISIBLE_DEVICES, else 0)."""
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])


def idle_closed(index, seconds):
    """Mean watts of the card over `seconds` before the device is opened."""
    with power.PowerSampler(index, idle_s=0) as meter:
        time.sleep(seconds)
    return meter.result.mean_w


# The card's nominal AICLK (MHz); cycles convert to seconds at this clock.
NOMINAL_MHZ = 1350


class ClockSampler:
    """Samples the card's AICLK (MHz) every `period_seconds` in a thread while in
    use."""

    def __init__(self, index, period_seconds=0.05):
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{index}/tt_aiclk")
        self.period_seconds, self.mhz = period_seconds, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.mhz.append(int(self.path.read_text()))
            self.stop.wait(self.period_seconds)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop.set()
        self.thread.join()

    def figures(self):
        """Minimum and mean MHz over the loop, and whether it left the nominal clock."""
        return {
            "aiclk_mhz_min": min(self.mhz),
            "aiclk_mhz_mean": float(np.mean(self.mhz)),
            "aiclk_samples": len(self.mhz),
            "aiclk_left_nominal": any(sample != NOMINAL_MHZ for sample in self.mhz),
        }


def power_figures(meter, windows, frames, idle_w):
    """The card's power over the windows in which commands ran."""
    source = meter.source
    parts = [
        power.summarize(
            meter.samples, window_start, window_end, source.update_s, source.limit_w
        )
        for window_start, window_end in windows
    ]
    seconds = sum(window_end - window_start for window_start, window_end in windows)
    if any(part.too_short for part in parts):
        return {"too_short": True, "note": parts[0].note, "seconds": seconds}
    joules = sum(part.joules for part in parts)
    mean = joules / seconds
    return {
        "card_idle_w": idle_w,
        "card_idle_open_w": meter.result.idle_mean_w,
        "card_mean_w": mean,
        "card_peak_w": max(part.peak_w for part in parts),
        "card_limit_w": source.limit_w,
        "card_mean_percent_of_limit": 100 * mean / source.limit_w,
        "card_joules_per_frame": joules / frames,
        "card_joules_per_frame_above_idle": (mean - idle_w) * seconds / frames,
        "seconds_per_frame": seconds / frames,
        "seconds": seconds,
        "whole_loop": meter.result.format(),
    }


def plan_loop(seconds, first_microseconds, max_repetitions, budget_seconds):
    """Commands for `seconds` of commands of `first_microseconds` each, at most
    `max_repetitions`. Exits when the loop would take longer than `budget_seconds`."""
    repetitions = int(
        min(max_repetitions, max(1, math.ceil(seconds * 1e6 / first_microseconds)))
    )
    planned = repetitions * first_microseconds * 1e-6
    card.log(f"plan: {repetitions} commands, about {planned:.1f} s on the card")
    if planned > budget_seconds:
        raise SystemExit(
            f"plan of {planned:.0f} s exceeds the budget of {budget_seconds} s"
        )
    return repetitions


def power_loop(index, running, handler, args, repetitions, batch=512):
    """`repetitions` commands of `handler` on every core in batches under the samplers;
    (meter, clock, the window in which commands ran, microseconds per command)."""
    latencies = []
    with power.PowerSampler(index) as meter, ClockSampler(index) as clock:
        start = time.perf_counter()
        while sum(map(len, latencies)) < repetitions:
            commands = min(batch, repetitions - sum(map(len, latencies)))
            latencies.append(card.loop(running, handler, args, commands, batch))
            if len(latencies[-1]) < commands:
                break
        # one command in flight at a time; the host's gaps between batches count too
        windows = [(start, time.perf_counter())]
    return meter, clock, windows, latencies


def loop_stats(windows, latencies):
    """Frame-time figures of a power loop."""
    latencies = np.concatenate(latencies)
    card_seconds = sum(
        window_end - window_start for window_start, window_end in windows
    )
    return {
        "frames": int(len(latencies)),
        "frame_microseconds_median": float(np.median(latencies)),
        "frame_microseconds_min": float(latencies.min()),
        "frame_microseconds_max": float(latencies.max()),
        "frames_per_s": len(latencies) / card_seconds,
        "card_s": card_seconds,
        "commands_s": float(latencies.sum()) * 1e-6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="tiny")
    parser.add_argument("--assign", default="balanced")
    parser.add_argument("--split", type=float, default=run.SPLIT, help="0: no bands")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-reps", type=int, default=100_000)
    parser.add_argument("--host-reps", type=int, default=5)
    parser.add_argument(
        "--budget", type=float, default=60.0, help="seconds the loop may take at most"
    )
    parser.add_argument("--json", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = {"name": f"card-{args.scene}", "scene": args.scene, "assign": args.assign}
    results["split"] = args.split
    scene, results["load_s"] = run.load(args.scene)
    index = card_index()
    idle_w = idle_closed(index, 3.0)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        results["host"], prepared = host_times(
            scene, workers, args.assign, args.host_reps, args.split
        )
        output_pages = len(prepared["tiles"]) * K["kPlanes"]
        uploaded = card.upload(
            session, prepared["pages"], prepared["directory"], output_pages
        )
        words = card.words(uploaded, K["kPlanes"])
        running = session.load(card.program(uploaded), deadline=1.0)
        first = card.loop(running, K["kHandlerFill"], words, 1)
        repetitions = plan_loop(
            args.seconds, float(first[0]), args.max_reps, args.budget
        )
        meter, clock, windows, latencies = power_loop(
            index, running, K["kHandlerFill"], words, repetitions
        )
        running.stop()
        results["run"] = card.check_run(
            running, workers, repetitions, np.concatenate(latencies)
        )
        started = time.perf_counter()
        frame = card.read_frame(session, uploaded)
        results["readback_s"] = time.perf_counter() - started
    results["loop"] = loop_stats(windows, latencies)
    results["power"] = power_figures(meter, windows, results["loop"]["frames"], idle_w)
    results["power"].update(clock.figures())
    results["frame_microseconds"] = {
        "first": float(first[0]),
        "median": results["loop"]["frame_microseconds_median"],
        "min": results["loop"]["frame_microseconds_min"],
        "max": results["loop"]["frame_microseconds_max"],
    }
    check = run.check(args.scene, None, *(prepared[name] for name in run.ARGS), frame)
    results["check"] = check
    results["ok"] = bool(results["run"]["ok"] and run.check_ok(check))
    print(json.dumps(results), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(results, file, indent=1)
    if args.out:
        run.save_result(args.out, results, frame, prepared["counts"], results["power"])
    if not results["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
