"""The thread-driven frame's parts per frame over `--runs` loads of `--frames` frames
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
from reference import raster

F = card_thread.F


def thread_runs(session, scene, uploaded, words, workers, args, index):
    """`args.runs` loads, each running `args.frames` frames; per run its result."""
    words = list(words)
    words[F["RF_W_REPEAT"]] = args.frames
    results, windows = [], []
    with power.PowerSampler(index) as meter, bench.ClockSampler(index) as clock:
        for run_number in range(args.runs):
            program = session.load(
                card_thread.program(scene, words, workers), deadline=1.0
            )
            result = program.thread_result(timeout=15.0)
            end = time.perf_counter()
            program.stop()
            status = card.check_run(program, workers, 0, [])
            entry = card_thread.result_dict(result)
            entry["frames"] = (list(result.words) + [0] * F["RF_OUT_WORDS"])[
                F["RF_O_FRAMES"]
            ]
            entry["status_ok"] = status["ok"]
            results.append(entry)
            windows.append((end - result.microseconds * 1e-6, end))
            card.log(
                f"run {run_number}: code {result.code}, {entry['frames']} frames,"
                f" {result.microseconds:.0f} us"
            )
            if result.code or not status["ok"]:
                break
    return results, windows, meter, clock


def per_frame(results):
    """Median over the runs of each part's microseconds per frame."""
    figures = {}
    for key in ("microseconds", "middle_microseconds"):
        names = results[0][key].keys()
        figures[key] = {
            part: float(
                np.median(
                    [result[key][part] / max(result["frames"], 1) for result in results]
                )
            )
            for part in names
        }
    figures["frame_microseconds"] = float(
        np.median(
            [
                result["thread_microseconds"] / max(result["frames"], 1)
                for result in results
            ]
        )
    )
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="s10k")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--frames", type=int, default=0, help="frames per run (0: auto)"
    )
    parser.add_argument(
        "--seconds", type=float, default=1.0, help="a run's length (auto)"
    )
    parser.add_argument(
        "--host", action="store_true", help="also one host-stepped frame"
    )
    parser.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="equal block partitions",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    report = {"scene": args.scene, "runs": args.runs, "frames_per_run": args.frames}
    scene, _ = run.load(args.scene)
    blocks = passes.mesh_blocks(scene)
    records = raster.setup(scene)
    index = bench.card_index()
    idle_w = bench.idle_closed(index, 3.0)
    with km.Session(log=card.log) as session:
        workers = card.workers(session)
        started = time.perf_counter()
        slots = 2 * len(scene["indices"]) + 65536
        uploaded = card_thread.tensors(session, blocks, workers, slots)
        report["upload_s"] = time.perf_counter() - started
        words = card_thread.words(
            uploaded, len(blocks), workers, workers, 4, args.balance
        )
        report["x280_mhz"] = session.pll_mhz
        if args.frames == 0:
            program = session.load(
                card_thread.program(scene, words, workers), deadline=1.0
            )
            first = program.thread_result(timeout=15.0)
            program.stop()
            if first.code:
                raise SystemExit(f"the cold frame ended with code {first.code}")
            args.frames = max(1, round(args.seconds * 1e6 / first.microseconds))
            report["frames_per_run"] = args.frames
            card.log(f"{args.frames} frames a run of about {args.seconds} s")
        results, windows, meter, clock = thread_runs(
            session, scene, uploaded, words, workers, args, index
        )
        report["thread"] = per_frame(results)
        report["thread"]["first_run"] = results[0]
        frames = sum(result["frames"] for result in results)
        report["power"] = bench.power_figures(meter, windows, frames, idle_w)
        report["power"].update(clock.figures())
        succeeded = all(
            result["code"] == 0 and result["status_ok"] for result in results
        )
        succeeded = succeeded and len(results) == args.runs
        started = time.perf_counter()
        image = card.read_frame(session, uploaded)
        report["read_frame_s"] = time.perf_counter() - started
        counts, offsets, members = raster.bin_triangles(records)
        tiles = list(range(lists.TILES))
        check = run.check(
            args.scene, None, records, offsets, members, tiles, counts, image
        )
        report["image_differs"] = check["any_differs"]
        succeeded = succeeded and run.check_ok(check)
        if args.host:
            host_arguments = argparse.Namespace(
                scene=args.scene, cores=0, setup=False, lists=-1
            )
            with bench.ClockSampler(index) as host_clock:
                host = card_passes.frame(host_arguments, session)
            report["host"] = {
                key: host[key]
                for key in (
                    "times",
                    "count_microseconds",
                    "setup_microseconds",
                    "fill_microseconds",
                    "heavy",
                    "items",
                )
                if key in host
            }
            report["host"]["clock"] = host_clock.figures()
            report["host"]["image_differs"] = host["check"]["any_differs"]
            succeeded = succeeded and host["ok"]
    report["ok"] = bool(succeeded)
    print(json.dumps(report), flush=True)
    if args.json:
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
