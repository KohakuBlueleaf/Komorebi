"""A session's own recovery, through public calls only: a handler that never returns on
one core, its give-up, the dispatcher's reset of that core, a reset by `reset_core`,
then MAP commands that must be exact. Prints one JSON line; see README.md."""

import json
import sys
import time

import numpy as np
from ladder import (
    HANDLER_SPIN,
    KERNELS,
    SPIN_KERNELS,
    SPIN_WHERE,
    STARTED,
    TILE,
    Maps,
    log,
)

import komorebi as km


def main():
    generator = np.random.default_rng(1)
    first, core, tiles, count = 14, 15, 120, 2
    group = (first, 2)
    report = {"core": core, "group": list(group)}
    with km.Session(log=log) as session:
        grid = session.grid
        cores = grid[0] * grid[1]
        everyone = (0, cores)
        # integers in [-2, 2]: every partial sum is exact in bfloat16
        values = generator.integers(-2, 3, (tiles * TILE, TILE)).astype(np.float32)
        source = session.tensor(values, "bfloat16", "tile")
        pages = (2 * count + 1) * (group[1] + cores)
        zeros = np.zeros((pages * TILE, TILE), np.float32)
        output = session.tensor(zeros, "bfloat16", "tile")
        program = km.Program(
            "mapcollect",
            kernels=SPIN_KERNELS,
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
        started = time.perf_counter()
        run = session.load(program, deadline=2.0, wait=4.0)  # recover=True by default
        report["load_s"] = round(time.perf_counter() - started, 3)
        report["recovery"] = run.recovery
        report["recovery_refused"] = run.recovery_refused
        log(f"loaded: recovery {run.recovery}, refused {run.recovery_refused}")
        maps = Maps(run, source, output, tiles)
        before = maps.batch([group, everyone], count)
        split = len(maps.done)
        log(f"SPIN in core {core}'s reader, no host action")
        spin = run.submit(HANDLER_SPIN, group, [1, SPIN_WHERE["reader"]])
        try:
            run.wait(spin, timeout=10.0)
            report["spin"] = "done"
        except km.CommandTimeout as timeout:
            report["spin"] = "given up" if timeout.given_up else "timeout"
            report["recovered"] = timeout.recovered
            report["recoveries"] = run.recoveries
            report["given_up_text"] = (
                run.recoveries[-1]["text"] if run.recoveries else None
            )
            log(f"timeout: {timeout}")
        log(f"reset_core({core}) of the healthy core")
        reset = run.reset_core(core)
        report["reset_core"] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in reset.items()
        }
        log(f"reset_core: {report['reset_core']}")
        after = maps.batch([group], 1)
        after += maps.batch([group, everyone], count)
        report["done_before"] = sum(event.kind == "done" for event in before)
        report["done_after"] = sum(event.kind == "done" for event in after)
        log("stop")
        run.stop()
        states = run.state()
        (got,) = session.read_bf16(output)
    checks = maps.exact(got.reshape(-1, TILE, TILE), values, maps.done)
    report["exact_before"] = all(checks[:split])
    report["exact_after"] = checks[split:]
    report["gave_up"] = [
        (index, kernel, state[kernel]["code"])
        for index, state in enumerate(states)
        for kernel in ("reader", "writer", "compute")
        if state[kernel]
    ]
    report["ok"] = bool(
        report["recovery"]
        and report.get("recovered") == [core]
        and report["given_up_text"]
        and "recovered" in report["given_up_text"]
        and report["reset_core"]["by"] == "dispatcher"
        and all(checks)
        and len(checks) > split
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report, default=str), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
