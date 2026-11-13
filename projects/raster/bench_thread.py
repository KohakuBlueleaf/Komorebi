"""The thread-conducted frame's parts per frame over `--runs` loads of `--frames` frames
(0: about `--seconds`), with clock and power over the threads' runs; `--host` adds a
host-stepped frame (`card_passes`) in the same session.
`python -m projects.raster.bench_thread --scene s10k [--runs 5] [--host]`"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import power
from projects.raster import bench, card, card_passes, card_thread, lists, passes, run
from ref import raster

F = card_thread.F


def thread_runs(session, scene, up, w, workers, args, index):
    """`args.runs` loads, each running `args.frames` frames; per run its result."""
    w = list(w)
    w[F["RF_W_REPEAT"]] = args.frames
    results, windows = [], []
    with power.PowerSampler(index) as meter, bench.ClockSampler(index) as clock:
        for i in range(args.runs):
            prog = session.load(card_thread.program(scene, w, workers), deadline=1.0)
            r = prog.thread_result(timeout=15.0)
            end = time.perf_counter()
            prog.stop()
            status = card.check_run(prog, workers, 0, [])
            res = card_thread.result_dict(r)
            res["frames"] = (list(r.words) + [0] * F["RF_OUT_WORDS"])[F["RF_O_FRAMES"]]
            res["status_ok"] = status["ok"]
            results.append(res)
            windows.append((end - r.us * 1e-6, end))
            card.log(f"run {i}: code {r.code}, {res['frames']} frames, {r.us:.0f} us")
            if r.code or not status["ok"]:
                break
    return results, windows, meter, clock


def per_frame(results):
    """Median over the runs of each part's microseconds per frame."""
    out = {}
    for key in ("us", "middle_us"):
        names = results[0][key].keys()
        out[key] = {
            n: float(np.median([r[key][n] / max(r["frames"], 1) for r in results]))
            for n in names
        }
    out["frame_us"] = float(
        np.median([r["thread_us"] / max(r["frames"], 1) for r in results])
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", default="s10k")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--frames", type=int, default=0, help="frames per run (0: auto)")
    p.add_argument("--seconds", type=float, default=1.0, help="a run's length (auto)")
    p.add_argument("--host", action="store_true", help="also one host-stepped frame")
    p.add_argument(
        "--no-balance", dest="balance", action="store_false", help="equal block shares"
    )
    p.add_argument("--json", default=None)
    args = p.parse_args()
    out = {"scene": args.scene, "runs": args.runs, "frames_per_run": args.frames}
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    recs = raster.setup(scene)
    index = bench.card_index()
    idle_w = bench.idle_closed(index, 3.0)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        t0 = time.perf_counter()
        slots = 2 * len(scene["indices"]) + 65536
        up = card_thread.tensors(session.device, blocks, workers, slots)
        out["upload_s"] = time.perf_counter() - t0
        w = card_thread.words(up, len(blocks), workers, workers, 4, args.balance)
        out["x280_mhz"] = session.pll_mhz
        if args.frames == 0:
            prog = session.load(card_thread.program(scene, w, workers), deadline=1.0)
            first = prog.thread_result(timeout=15.0)
            prog.stop()
            if first.code:
                raise SystemExit(f"the first frame ended with code {first.code}")
            args.frames = max(1, round(args.seconds * 1e6 / first.us))
            out["frames_per_run"] = args.frames
            card.log(f"{args.frames} frames a run of about {args.seconds} s")
        results, windows, meter, clock = thread_runs(
            session, scene, up, w, workers, args, index
        )
        out["thread"] = per_frame(results)
        out["thread"]["first_run"] = results[0]
        frames = sum(r["frames"] for r in results)
        out["power"] = bench.power_figures(meter, windows, frames, idle_w)
        out["power"].update(clock.figures())
        ok = all(r["code"] == 0 and r["status_ok"] for r in results)
        ok = ok and len(results) == args.runs
        t0 = time.perf_counter()
        img = card.read_frame(session, up)
        out["read_frame_s"] = time.perf_counter() - t0
        counts, offsets, members = raster.bin_triangles(recs)
        tiles = list(range(lists.TILES))
        check = run.check(args.scene, None, recs, offsets, members, tiles, counts, img)
        out["image_differs"] = check["any_differs"]
        ok = ok and run.check_ok(check)
        if args.host:
            hargs = argparse.Namespace(scene=args.scene, cores=0, setup=False, lists=-1)
            with bench.ClockSampler(index) as hclock:
                host = card_passes.frame(hargs, session)
            out["host"] = {
                k: host[k]
                for k in ("times", "count_us", "setup_us", "fill_us", "heavy", "items")
                if k in host
            }
            out["host"]["clock"] = hclock.figures()
            out["host"]["image_differs"] = host["check"]["any_differs"]
            ok = ok and host["ok"]
    out["ok"] = bool(ok)
    print(json.dumps(out), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
