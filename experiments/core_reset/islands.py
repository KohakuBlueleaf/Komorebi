"""The session's recovery on two cards as islands, public calls only: a handler that
never returns on one core of island 1 (--step 1: in a group of island 1's ranks; 2: in
one group over every rank), its give-up, island 1's dispatcher resetting that core, a
MAP on the group that must be exact, a clean stop. One JSON line; see README.md."""

import argparse
import json
import sys
import time

import numpy as np
from ladder import HANDLER_SPIN, KERNELS, SPIN_KERNELS, SPIN_WHERE, STARTED, TILE, log

import komorebi as km
from komorebi.layout import partition

NO_RANK = 0xFFFFFFFF


def map_words(source, output, group):
    """MAP arguments of group (first, count): rank r adds its part of the group's
    2 x count input tiles from tile 2 x first, into page first + r."""
    first, count = group
    return [source.address, 2 * count, output.address, first, 0, NO_RANK, 2 * first]


def exact(got, values, group):
    """Each rank's page of a MAP on `group`: the sum of its part of the tiles."""
    tiles = values.reshape(-1, TILE, TILE).astype(np.float64)
    first, count = group
    return all(
        np.array_equal(
            got[first + rank].astype(np.float64),
            tiles[
                [2 * first + t for t in partition(range(2 * count), rank, count)]
            ].sum(0),
        )
        for rank in range(count)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(1, 2), required=True)
    step = parser.parse_args().step
    generator = np.random.default_rng(1)
    report = {"cards": [0, 1], "step": step}
    with km.Session(cards=[0, 1], log=log) as session:
        ranks = session.ranks
        graph = session.graph
        island, core = 1, 15  # not a column leader, not core 0
        spinning = graph.first(island) + core
        group = (graph.first(island), graph.count(island)) if step == 1 else (0, ranks)
        report.update({"ranks": ranks, "group": group, "island": island, "core": core})
        values = generator.integers(-2, 3, (2 * ranks * TILE, TILE)).astype(np.float32)
        source = session.tensor(values, "bfloat16", "tile")
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
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
        run = session.load(program, deadline=2.0, wait=4.0)  # recover=True by default
        report["recovery"] = [
            kept and [len(kept["lists"]), kept["bytes"], round(kept["seconds"], 4)]
            for kept in run.recovery
        ]  # per island: lists, bytes, seconds
        log(f"loaded: recovery {report['recovery']}")
        try:  # every rank of island 1
            report["inspect"] = len(
                run.inspect((graph.first(island), graph.count(island)))
            )
        except km.KomorebiError as error:
            report["inspect"] = str(error)
        log(f"inspect of island {island}: {report['inspect']}")
        handler = run.program.handler("map")
        before = run.wait(run.submit(handler, group, map_words(source, output, group)))
        report["before"] = [event.kind for event in before]
        log(f"SPIN on rank {spinning} (island {island} core {core}), no host action")
        where = [spinning - group[0], SPIN_WHERE["reader"]]  # its index in the group
        spin = run.submit(HANDLER_SPIN, group, where)
        try:
            report["spin"] = [event.kind for event in run.wait(spin, timeout=15.0)]
        except km.CommandTimeout as timeout:
            report["spin"] = "given up" if timeout.given_up else "timeout"
            report["timeout_island"] = timeout.island
            report["recovered"] = timeout.recovered
            log(f"timeout: {timeout}")
            if not timeout.given_up:
                report["error"] = str(timeout)
                print(json.dumps(report, default=str), flush=True)
                sys.exit(1)
        report["recoveries"] = run.recoveries
        log(f"recoveries: {run.recoveries}")
        after = run.wait(run.submit(handler, group, map_words(source, output, group)))
        report["after"] = [event.kind for event in after]
        log(f"MAP on {group} after: {report['after']}")
        log("stop")
        try:
            results = run.stop()
        except km.KomorebiError as error:
            report["stop"] = str(error)
            print(json.dumps(report, default=str), flush=True)
            sys.exit(1)
        report["stop"] = "clean"
        report["kernel_errors"] = [result["error_name"] for result in results]
        (got,) = session.read_bf16(output)
    report["exact_after"] = exact(got.reshape(-1, TILE, TILE), values, group)
    report["ok"] = bool(
        report["spin"] == "given up"
        and report.get("timeout_island") == island
        and report.get("recovered") == [core]
        and [(entry["island"], entry["cores"]) for entry in report["recoveries"]]
        == [(island, [core])]
        and report["after"] == ["done"] * step  # one part an island
        and report["stop"] == "clean"
        and isinstance(report["inspect"], int)
        and report["exact_after"]
    )
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report, default=str), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
