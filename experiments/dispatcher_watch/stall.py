"""The dispatcher's watchdog: core 0's loop stalls on a debug command, core 2 reports
it and interrupts core 0, which re-enters its loop; MAP commands after must be exact.
Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import native
from komorebi.layout import partition

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "01_mapcollect" / "kernels"
TILE = 32
TICKS_S = 50e6  # the X280's mtime
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stall-ms", type=int, default=3000, help="the stall's bound")
    parser.add_argument("--tiles", type=int, default=120, help="input tiles of a MAP")
    parser.add_argument("--maps", type=int, default=2, help="MAPs of each kind")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def maps(run, source, output, tiles, groups, count, page):
    """`count` MAPs on each group into their own pages; their events and pages."""
    handles, placed = [], []
    for _ in range(count):
        for group in groups:
            words = [source.address, tiles, output.address, page, 0, 0xFFFFFFFF, 0]
            handles.append(run.submit("map", group, words))
            placed.append((page, group[1]))
            page += group[1]
    return run.wait(handles, timeout=10.0), placed, page


def exact(got, values, tiles, placed):
    """Each MAP's pages: each rank's sum of its partition of the tiles."""
    blocks = values.reshape(-1, TILE, TILE).astype(np.float64)
    return [
        all(
            np.array_equal(
                got[first + rank].astype(np.float64),
                blocks[partition(range(tiles), rank, count)].sum(0),
            )
            for rank in range(count)
        )
        for first, count in placed
    ]


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    report = {"stall_ms": args.stall_ms}
    with km.Session(log=log) as session:
        grid = session.grid
        cores = grid[0] * grid[1]
        groups = [(14, 2), (0, cores)]
        # integers in [-2, 2]: every partial sum is exact in bfloat16
        values = generator.integers(-2, 3, (args.tiles * TILE, TILE)).astype(np.float32)
        source = session.tensor(values, "bfloat16", "tile")
        pages = 2 * args.maps * (2 + cores)
        output = session.tensor(
            np.zeros((pages * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": source.accessor_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            depth=4,
            handlers=KERNELS / "mapcollect.h",
        )
        image = native.image_path(session.tree, defines=["KMD_STALL_DEBUG=1"])
        run = session.load(program, deadline=2.0, wait=4.0, image=str(image))
        log(f"loaded, grid {grid}, image {image.parent.name}; MAPs before the stall")
        before, placed, page = maps(
            run, source, output, args.tiles, groups, args.maps, 0
        )
        split = len(placed)
        handle = run._queue_command(
            run.h["KMD_HANDLER_STALL"],
            (0, 1),
            [args.stall_ms],
            flags=run.h["KMD_FLAG_DISPATCHER"],
        )
        run._counts[-1] = 0  # the kernel's own command: no sequence number
        (pong,) = run.wait([handle], timeout=10.0)
        stalled = time.perf_counter()
        log(f"stall command answered: {pong}; no host action for 0.5 s")
        time.sleep(0.5)
        log("MAPs after the stall")
        after, more, page = maps(
            run, source, output, args.tiles, groups, args.maps, page
        )
        placed += more
        report["host_s"] = round(time.perf_counter() - stalled, 3)
        report["stalls"] = run.stalls
        log(f"stall events: {run.stalls}")
        report["done_before"] = sum(event.kind == "done" for event in before)
        report["done_after"] = sum(event.kind == "done" for event in after)
        log("stop")
        result = run.stop()
        (got,) = session.read_bf16(output)
    h = run.h
    words = result["x280_cores"]
    report["x280_cores"] = {
        name: words[h[f"KMD_H_{name.upper()}"]]
        for name in ("watch", "stalls", "back", "watch_gap", "stall_pc", "stall_passes")
    }
    ticks = {event["stage"]: event["ticks"] for event in run.stalls}
    if "stalled" in ticks and "back" in ticks:
        report["stall_to_back_ms"] = round(
            (ticks["stalled"] + ticks["back"]) / TICKS_S * 1e3, 3
        )
        report["seen_to_back_us"] = round(ticks["back"] / TICKS_S * 1e6, 1)
    report["longest_gap_ms"] = round(words[h["KMD_H_WATCH_GAP"]] / TICKS_S * 1e3, 3)
    checks = exact(got.reshape(-1, TILE, TILE), values, args.tiles, placed)
    report["exact_before"] = all(checks[:split])
    report["exact_after"] = checks[split:]
    report["ok"] = bool(
        all(checks)
        and [event["stage"] for event in run.stalls] == ["stalled", "back"]
        and report.get("stall_to_back_ms", 1e9) < args.stall_ms
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
