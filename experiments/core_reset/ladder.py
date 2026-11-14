"""One core of a run reset and loaded again while the other cores keep their kernels:
MAP commands before and after, on the core's group and on every core, must be exact.
Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.layout import partition

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parents[1] / "examples" / "01_mapcollect" / "kernels"
TILE = 32
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--step", type=int, default=1, help="1: by the host, 2: by the dispatcher"
    )
    parser.add_argument("--first", type=int, default=14, help="the group's first core")
    parser.add_argument("--core", type=int, default=15, help="the core reset")
    parser.add_argument("--tiles", type=int, default=120, help="input tiles of a MAP")
    parser.add_argument("--maps", type=int, default=2, help="MAPs of each kind")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


class Maps:
    """MAP commands, each into its own output pages, and the check of every page."""

    def __init__(self, run, source, output, tiles):
        self.run, self.source, self.output, self.tiles = run, source, output, tiles
        self.page = 0
        self.look = None  # a group to inspect while the commands are in flight
        self.done = []  # (first page, count)

    def submit(self, group):
        words = [self.source.address, self.tiles, self.output.address, self.page, 0]
        handle = self.run.submit("map", group, words + [0xFFFFFFFF, 0])
        self.done.append((self.page, group[1]))
        self.page += group[1]
        return handle

    def batch(self, groups, count):
        """`count` MAPs on each group; the events of all, or the timeout's inspect."""
        handles = [self.submit(group) for _ in range(count) for group in groups]
        if self.look is not None:
            time.sleep(0.3)
            for core in self.run.inspect(self.look):
                log(f"inspect in flight: {brief(core)}")
        try:
            return self.run.wait(handles, timeout=10.0)
        except km.CommandTimeout as timeout:
            log(f"timeout: given_up {timeout.given_up}")
            for core in timeout.cores or []:
                log(f"at the deadline: {brief(core)}")
            return []

    def exact(self, got, values, maps):
        """Each listed MAP's pages: each rank's sum of its partition of the tiles."""
        tiles = values.reshape(-1, TILE, TILE).astype(np.float64)
        out = []
        for first, count in maps:
            out.append(
                all(
                    np.array_equal(
                        got[first + rank].astype(np.float64),
                        tiles[partition(range(self.tiles), rank, count)].sum(0),
                    )
                    for rank in range(count)
                )
            )
        return out


def brief(core):
    """An inspect's view of one core, without its trace, as one line."""
    skip = ("trace", "last", "reported")
    return json.dumps({k: v for k, v in core.items() if k not in skip}, default=str)


def dispatcher_reset(run, core, x, y, report):
    """The dispatcher resets `core`: the host uploads the write list and sends the
    command; the X280's ticks and the host's seconds. True: the core is back."""
    started = time.perf_counter()
    writes = run.backend.launch.reset_list(x, y)
    run.backend.port.upload(0, writes)
    report["list_bytes"] = len(writes)
    report["list_s"] = round(time.perf_counter() - started, 4)
    log(f"write list: {len(writes)} B uploaded in {report['list_s']} s")
    started = time.perf_counter()
    handle = run._queue_command(run.h["KMD_HANDLER_RESET"], (core, 1), [0, len(writes)])
    run._counts[-1] = 0  # the kernel's own command: no sequence number
    (event,) = run.wait([handle], timeout=10.0)
    report["reset_s"] = round(time.perf_counter() - started, 4)
    report["x280_s"] = round(event.ticks / 50e6, 4)
    report["status"] = event.auxiliary
    log(f"reset by the dispatcher: {event}, host {report['reset_s']} s")
    return event.kind == "pong" and event.auxiliary == run.h["KMD_RESET_OK"]


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    group = (args.first, 2)
    report = {"step": args.step, "group": list(group), "core": args.core}
    with km.Session(log=log) as session:
        grid = session.grid
        cores = grid[0] * grid[1]
        everyone = (0, cores)
        # integers in [-2, 2]: every partial sum is exact in bfloat16
        values = generator.integers(-2, 3, (args.tiles * TILE, TILE)).astype(np.float32)
        source = session.tensor(values, "bfloat16", "tile")
        pages = (2 * args.maps + 1) * (group[1] + cores)
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
                "reader": source.accessor_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            depth=4,
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program, deadline=2.0, wait=4.0)
        maps = Maps(run, source, output, args.tiles)
        log(f"loaded, grid {grid}; MAPs before the reset")
        before = maps.batch([group, everyone], args.maps)
        split = len(maps.done)
        x, y = args.core % grid[0], args.core // grid[0]
        if args.step == 1:
            log(f"reset of core {args.core} ({x}, {y}) by the host")
            started = time.perf_counter()
            parts = run.backend.launch.reset_core(x, y, 2.0)
            report["reset_s"] = round(time.perf_counter() - started, 4)
            report["parts_s"] = [round(part, 4) for part in parts]
            log(f"reset done: {report['reset_s']} s, parts {report['parts_s']}")
            back = parts[1] >= 0
        else:
            back = dispatcher_reset(run, args.core, x, y, report)
        time.sleep(0.1)
        for core in run.inspect(group):
            log(f"inspect after the reset: {brief(core)}")
        if not back:
            report["firmware"] = False
            log("the core's firmware did not report DONE")
        else:
            log("MAPs after the reset")
            maps.look = group
            after = maps.batch([group], 1)
            maps.look = None
            after += maps.batch([group, everyone], args.maps)
            report["done_after"] = sum(event.kind == "done" for event in after)
        report["done_before"] = sum(event.kind == "done" for event in before)
        log("stop")
        if not run.stopped:
            run.stop()
        states = run.state()
        log("read back")
        (got,) = session.read_bf16(output)
    got = got.reshape(-1, TILE, TILE)
    checks = maps.exact(got, values, maps.done)
    report["exact_before"] = all(checks[:split])
    report["exact_after"] = checks[split:]
    report["gave_up"] = [
        (core, kernel, state[kernel]["code"])
        for core, state in enumerate(states)
        for kernel in ("reader", "writer", "compute")
        if state[kernel]
    ]
    report["ok"] = bool(
        report["exact_before"] and checks[split:] and all(checks[split:])
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
