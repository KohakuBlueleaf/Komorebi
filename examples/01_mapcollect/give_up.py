"""HANG commands the dispatcher gives up at their deadline while the run goes on: for
each wait that never ends, an inspect names it on the hung core, the command's timeout
says it was given up, and MAP commands on the same cores are exact. Then the cost of an
inspect of every core. Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.layout import partition
from komorebi.state import ROLES

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
STARTED = time.monotonic()
# hang_mode -> (name, the kernel that waits, the wait an inspect names)
MODES = {
    1: ("port", "reader", "port"),
    2: ("input", "unpack", "cinput"),
    3: ("output", "writer", "data"),
}


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=2, help="group size (2 or more)")
    parser.add_argument("--tiles", type=int, default=8, help="input tiles of a MAP")
    parser.add_argument("--maps", type=int, default=3, help="MAPs after each give-up")
    parser.add_argument("--deadline", type=float, default=0.5, help="seconds")
    parser.add_argument("--rounds", type=int, default=50, help="rounds of the cost")
    parser.add_argument("--modes", default="port,input,output", help="hangs, by name")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def wheres(cores, rank):
    """Each kernel's wait on the core of rank `rank` of an inspect's list."""
    if rank >= len(cores) or not cores[rank]["read"]:
        return None
    live = cores[rank]["live"]
    return {role: live[role]["where"] for role in ROLES}


def main():
    args = parse()
    names = args.modes.split(",")
    modes = {mode: MODES[mode] for mode in MODES if MODES[mode][0] in names}
    generator = np.random.default_rng(args.seed)
    group = (args.first, args.count)
    report = {"group": list(group), "deadline_s": args.deadline, "modes": {}}
    with km.Session(log=log) as session:
        # integers in [-2, 2]: every partial sum is exact in bfloat16
        values = generator.integers(-2, 3, (args.tiles * TILE, TILE)).astype(np.float32)
        source = session.tensor(values, "bfloat16", "tile")
        pages = max(len(modes) * args.maps * args.count, 1)
        zeros = np.zeros((pages * TILE, TILE), np.float32)
        output = session.tensor(zeros, "bfloat16", "tile")
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": source.accessor_args() + 2 * source.placement_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            depth=4,
            handlers=KERNELS / "mapcollect.h",
        )
        commands = len(modes) * (1 + args.maps) + 4 * 2 * args.rounds
        run = session.load(
            program, trace=commands + 1, deadline=args.deadline, wait=4 * args.deadline
        )
        log(f"loaded, HANG on rank 1 of cores {group}")
        page = 0
        for mode, (name, role, where) in modes.items():
            hang = run.submit("hang", group, [1, mode])
            time.sleep(args.deadline / 4)
            inspected = run.inspect(group)
            try:
                run.wait(hang, timeout=10.0)
                error = None
            except km.CommandTimeout as timeout:
                error = timeout
            maps = []
            for _ in range(args.maps):
                words = [source.address, args.tiles, output.address, page, 0]
                maps.append(run.submit("map", group, words + [0xFFFFFFFF, 0]))
                page += args.count
            events = run.wait(maps, timeout=10.0)
            report["modes"][name] = {
                "inspect": wheres(inspected, 1),
                "inspect_others": wheres(inspected, 0),
                "trace_entries": len(inspected[1]["trace"]),
                "given_up": bool(error and error.given_up),
                "at_deadline": (
                    wheres(error.cores, 1) if error and error.cores else None
                ),
                "maps_done": sum(event.kind == "done" for event in events),
                "kernel": role,
                "expected": where,
            }
            log(f"{name}: {report['modes'][name]}")
        report["inspect"] = cost(run, args)
        run.stop()
        states = run.state()
        (got,) = session.read_bf16(output)
    report["maps_exact"] = exact(got.reshape(-1, TILE, TILE)[:page], values, args)
    report["deadline_records"] = sum(
        1
        for state in states
        for kernel in ("reader", "writer", "compute")
        if state[kernel] and state[kernel]["code"] == "deadline"
    )
    report["ok"] = bool(
        report["maps_exact"]
        and report["inspect"]["read"] == report["inspect"]["cores"]
        and all(
            mode["given_up"]
            and mode["inspect"][mode["kernel"]] == mode["expected"]
            and mode["maps_done"] == args.maps
            for mode in report["modes"].values()
        )
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def cost(run, args):
    """An inspect of every core: the dispatcher's microseconds and the host's; then the
    microseconds of four NOP commands in flight, alone and with an inspect behind them.
    """
    started = time.perf_counter()
    cores = run.inspect()
    host = (time.perf_counter() - started) * 1e6
    out = {"cores": len(cores), "read": sum(core["read"] for core in cores)}
    out["waiting_for_a_command"] = sum(
        wheres(cores, rank)["reader"] == "next" for rank in range(len(cores))
    )
    out |= {"microseconds": run.inspect_ticks / 50, "host_microseconds": round(host)}
    groups = [(args.first + args.count + 2 * index, 2) for index in range(4)]
    for kind in ("alone", "with_inspect"):
        took = []
        for _ in range(args.rounds):
            handles = [run.submit(0, group) for group in groups]
            if kind == "with_inspect":
                run.inspect()
            took += [event.microseconds for event in run.wait(handles)]
        out[kind] = round(float(np.mean(took)), 2)
    return out


def exact(got, values, args):
    """Every MAP's tiles: each rank's sum of its partition of the input tiles."""
    tiles = values.reshape(-1, TILE, TILE).astype(np.float64)
    want = [
        tiles[partition(range(args.tiles), rank, args.count)].sum(0)
        for rank in range(args.count)
    ]
    return all(
        np.array_equal(got[page].astype(np.float64), want[page % args.count])
        for page in range(len(got))
    )


if __name__ == "__main__":
    main()
