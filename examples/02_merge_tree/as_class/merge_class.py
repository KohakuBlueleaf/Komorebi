"""The merge tree written as a class: MERGE of ../merge.py as one method, its tree of
arrivals one km.reduce line, compiled into the program's files (build/mergetree-<hash>/
beside this file); the same jobs and checks as ../merge.py. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power
from komorebi.dsl.program import lines_of

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import merge as hand  # noqa: E402  (the hand program's jobs and checks)
import merge_fed  # noqa: E402

TILE = hand.TILE


class MergeTree(km.Program):
    """Each core of the group adds its part of `tiles` input tiles from `first_tile`;
    the group's parts are summed by owner (form 0) or along a tree (form 1) and the
    leader writes the sum at output page `first_page`."""

    @km.handler
    def merge(
        self,
        source: km.Tensor,
        tiles: int,
        output: km.Tensor,
        first_page: int,
        form: int,
        first_tile: int,
    ):
        part = self.partition(source.pages(first_tile, tiles))
        mine = km.tile.zeros(km.fp32)
        for tile in km.Stream(source[part], block=8):
            mine += tile
        whole = km.reduce(mine, form=form)
        if self.rank == self.leader:
            # the result word: the sum's 1,024 32-bit words added (merge.py's checksum)
            checksum = km.cxx(
                """
                const volatile tt_l1_ptr uint32_t* words =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(whole));
                for (uint32_t word = 0; word < 1024; word++) {
                    checksum += words[word];
                }
                """,
                takes=[whole],
                gives=int,
                on="mover_b",
            )
            output.pages(first_page, 1).write(whole)
            return checksum


def launch(session, args, jobs, inputs, output):
    """One load of the class's program: the jobs (each writes output page job mod
    pages), then the stop and the state, as the hand program's launch reports them."""
    program = MergeTree(source=inputs[0], output=output, depth=args.depth, log=hand.log)
    run = session.load(program, deadline=args.deadline, wait=args.wait)
    hand.log(f"loaded, {len(jobs)} jobs")
    handles = []
    start = time.perf_counter()
    try:
        for job, (first, count, form, source) in enumerate(jobs):
            arguments = (inputs[source % 2], args.tiles, output, job % args.out_pages)
            handles.append(
                run.submit(
                    MergeTree.merge,
                    (first, count),
                    args=(*arguments, form, source),
                    exclusive=True,
                )
            )
        hand.log(f"{run.drain(timeout=30.0)} events")
    except km.CommandTimeout as error:
        hand.log(f"timeout: {error}")
        for line in lines_of(program, error):
            hand.log(f"  {line}")
        raise
    host_seconds = time.perf_counter() - start
    run.stop()
    events = [run.event(handle) for handle in handles]
    states = run.state()
    summary = {
        "kernel": {
            key: run.result()[key] for key in ("error_name", "commands", "events")
        },
        "kernel_text": program.sizes.text,
        "done": sum(event is not None and event.kind == "done" for event in events),
        "microseconds_per_job_host": round(host_seconds * 1e6 / len(jobs), 3),
        "cores_ended": {
            key: sum(state[key] != 0 for state in states)
            for key in ("reader_end", "writer_end", "compute_end")
        },
        "compute_wrong": sum(sum(state["compute_wrong"]) for state in states),
        "gave_up": hand.gave_up(states)[:8],
        "waiting_ports": {
            state["core"]: {
                port: port_state["want"] - port_state["count"]
                for port, port_state in enumerate(state["ports"])
                if port_state["want"]
            }
            for state in states
            if any(port_state["want"] for port_state in state["ports"])
        },
    }
    return events, summary


def run(session, args, generator):
    """The hand program's run() with the class's program: the same jobs and inputs."""
    grid_width, grid_height = session.grid
    jobs = merge_fed.make_jobs(
        args,
        min(grid_width * grid_height, hand.CONSTANTS["kArrivePages"] // args.part + 1),
        generator,
    )
    rows = (args.tiles + max(job[3] for job in jobs) + 1) * TILE
    input_arrays = [
        generator.integers(-2, 3, (rows, TILE)).astype(np.float32),
        generator.uniform(-1, 1, (rows, TILE)).astype(np.float32),
    ]

    def tiles(array):
        return session.tensor(array, "float32", "tile")

    inputs = [tiles(array) for array in input_arrays]
    output = tiles(np.zeros((args.out_pages * TILE, TILE), dtype=np.float32))
    events, summary = launch(session, args, jobs, inputs, output)
    result = {
        "jobs": len(jobs),
        "tiles": args.tiles,
        "form": args.form,
        **summary,
        **hand.checked(args, jobs, events, output, input_arrays),
    }
    result["ok"] = hand.verdict(result, len(jobs), grid_width * grid_height)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    ours, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = hand.parse()
    if args.part != 1 or args.stall_rank >= 0:
        raise SystemExit("the class sums parts of one tile and stalls no rank")
    report = {"runs": {}}
    with km.Session(log=hand.log) as session:
        hand.log("device open")
        if ours.beside:
            report["runs"]["hand"] = hand.run(
                session, args, np.random.default_rng(args.seed)
            )
            hand.log(f"hand program: ok {report['runs']['hand']['ok']}")
        report["runs"]["class"] = run(session, args, np.random.default_rng(args.seed))
        hand.log(f"class: ok {report['runs']['class']['ok']}")
    report["ok"] = all(each["ok"] for each in report["runs"].values())
    report["wall_s"] = round(time.monotonic() - hand.STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
