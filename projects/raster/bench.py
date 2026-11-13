"""Frame time and power of the rasterizer on a card: host steps, then FILL records back
to back in one run under the power and clock samplers, power over the stretches the
records ran (see the README's "Time"). `--out` writes the results layout.
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


def host_times(scene, workers, mode, reps, split):
    """Median seconds of each host step over `reps` preparations, and the last one."""
    steps, h = [], None
    for _ in range(reps):
        h = run.prepare(scene, workers, mode, split=split)
        steps.append(h["host"])
    return {k: float(np.median([s[k] for s in steps])) for k in steps[0]}, h


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
    """Samples the card's AICLK (MHz) every `period_s` in a thread while in use."""

    def __init__(self, index, period_s=0.05):
        self.path = Path(f"/sys/class/tenstorrent/tenstorrent!{index}/tt_aiclk")
        self.period_s, self.mhz = period_s, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            self.mhz.append(int(self.path.read_text()))
            self.stop.wait(self.period_s)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def figures(self):
        """Minimum and mean MHz over the loop, and whether it left the nominal clock."""
        return {
            "aiclk_mhz_min": min(self.mhz),
            "aiclk_mhz_mean": float(np.mean(self.mhz)),
            "aiclk_samples": len(self.mhz),
            "aiclk_left_nominal": any(m != NOMINAL_MHZ for m in self.mhz),
        }


def power_figures(meter, windows, frames, idle_w):
    """The card's power over the windows in which records ran."""
    src = meter.source
    parts = [
        power.summarize(meter.samples, a, b, src.update_s, src.limit_w)
        for a, b in windows
    ]
    seconds = sum(b - a for a, b in windows)
    if any(p.too_short for p in parts):
        return {"too_short": True, "note": parts[0].note, "seconds": seconds}
    joules = sum(p.joules for p in parts)
    mean = joules / seconds
    return {
        "card_idle_w": idle_w,
        "card_idle_open_w": meter.result.idle_mean_w,
        "card_mean_w": mean,
        "card_peak_w": max(p.peak_w for p in parts),
        "card_limit_w": src.limit_w,
        "card_mean_percent_of_limit": 100 * mean / src.limit_w,
        "card_joules_per_frame": joules / frames,
        "card_joules_per_frame_above_idle": (mean - idle_w) * seconds / frames,
        "seconds_per_frame": seconds / frames,
        "seconds": seconds,
        "whole_loop": meter.result.format(),
    }


def plan_loop(seconds, first_us, max_reps, budget_s):
    """Records for `seconds` of records of `first_us` each, at most `max_reps`. Exits
    when the loop would take longer than `budget_s`."""
    reps = int(min(max_reps, max(1, math.ceil(seconds * 1e6 / first_us))))
    planned = reps * first_us * 1e-6
    card.log(f"plan: {reps} records, about {planned:.1f} s on the card")
    if planned > budget_s:
        raise SystemExit(f"plan of {planned:.0f} s exceeds the budget of {budget_s} s")
    return reps


def power_loop(index, run_, seg, args, reps, batch=512):
    """`reps` records of `seg` on every core in batches under the samplers; (meter,
    clock, the window in which records ran, microseconds per record)."""
    lat = []
    with power.PowerSampler(index) as meter, ClockSampler(index) as clock:
        start = time.perf_counter()
        while sum(map(len, lat)) < reps:
            n = min(batch, reps - sum(map(len, lat)))
            lat.append(card.loop(run_, seg, args, n, batch))
            if len(lat[-1]) < n:
                break
        # one record in flight at a time; the host's gaps between batches count too
        windows = [(start, time.perf_counter())]
    return meter, clock, windows, lat


def loop_stats(windows, lat):
    """Frame-time figures of a power loop."""
    lat = np.concatenate(lat)
    card_s = sum(b - a for a, b in windows)
    return {
        "frames": int(len(lat)),
        "frame_us_median": float(np.median(lat)),
        "frame_us_min": float(lat.min()),
        "frame_us_max": float(lat.max()),
        "frames_per_s": len(lat) / card_s,
        "card_s": card_s,
        "records_s": float(lat.sum()) * 1e-6,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="tiny")
    p.add_argument("--assign", default="balanced")
    p.add_argument("--split", type=float, default=run.SPLIT, help="0: no bands")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--max-reps", type=int, default=100_000)
    p.add_argument("--host-reps", type=int, default=5)
    p.add_argument(
        "--budget", type=float, default=60.0, help="seconds the loop may take at most"
    )
    p.add_argument("--json", default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = {"name": f"card-{args.scene}", "scene": args.scene, "assign": args.assign}
    out["split"] = args.split
    scene, out["load_s"] = run.load(args.scene)
    index = card_index()
    idle_w = idle_closed(index, 3.0)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        out["host"], h = host_times(
            scene, workers, args.assign, args.host_reps, args.split
        )
        n_out = len(h["tiles"]) * K["kPlanes"]
        up = card.upload(session, h["pages"], h["directory"], n_out)
        words = card.words(up, K["kPlanes"])
        run_ = session.load(card.program(), deadline=1.0)
        first = card.loop(run_, K["kSegFill"], words, 1)
        reps = plan_loop(args.seconds, float(first[0]), args.max_reps, args.budget)
        meter, clock, windows, lat = power_loop(index, run_, K["kSegFill"], words, reps)
        run_.stop()
        out["run"] = card.check_run(run_, workers, reps, np.concatenate(lat))
        t0 = time.perf_counter()
        frame = card.read_frame(session, up)
        out["readback_s"] = time.perf_counter() - t0
    out["loop"] = loop_stats(windows, lat)
    out["power"] = power_figures(meter, windows, out["loop"]["frames"], idle_w)
    out["power"].update(clock.figures())
    out["frame_us"] = {
        "first": float(first[0]),
        "median": out["loop"]["frame_us_median"],
        "min": out["loop"]["frame_us_min"],
        "max": out["loop"]["frame_us_max"],
    }
    check = run.check(args.scene, None, *(h[k] for k in run.ARGS), frame)
    out["check"] = check
    out["ok"] = bool(out["run"]["ok"] and run.check_ok(check))
    print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    if args.out:
        run.save_result(args.out, out, frame, h["counts"], out["power"])
    if not out["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
