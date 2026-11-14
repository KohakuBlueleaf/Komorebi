"""The matrix product over the islands of a session: A and Y split by row tiles into
even parts by island, W a copy on every island; each island runs the grid plan of its
own rows, one GRID command a call over every island's cores (one card: every row).
Prints one JSON line; exits 1 when the check fails. Usage in README.md."""

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

G = pl.G
log = run.log


def island_rows(a_tensor, k_tiles):
    """[(first row tile, row tiles)] of each island: its part of A's pages."""
    return [
        (first // k_tiles, pages // k_tiles)
        for first, pages in a_tensor.placement.parts
    ]


class Rows:
    """The product loaded on every island: island i's grid plan on its rows of A and
    Y, one GRID command a call on the flat group over every island's rectangle,
    exclusive (its broadcasts)."""

    def __init__(self, session, dimensions, tensors, wait=1.0):
        graph, sessions = session.graph, session.sessions
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.tensors = tensors
        self.rows = island_rows(tensors["a"], k_tiles)
        if any(rows == 0 for _, rows in self.rows):
            raise SystemExit(f"{m_tiles} row tiles leave an island without rows")
        nocs = [one.nocs() for one in sessions]
        self.plans = [
            pl.grid_plan(rows, k_tiles, n_tiles, grid=island_nocs.grid)
            for (_, rows), island_nocs in zip(self.rows, nocs)
        ]
        if None in self.plans:
            raise SystemExit(f"no grid plan fits L1 for {dimensions}")
        last = len(self.plans) - 1
        rect = self.plans[last].group(nocs[last].grid)
        self.group = (0, graph.first(last) + rect[0] + rect[1])
        word = self.group[0] | self.group[1] << 16
        reached = {
            island: (first, count) for island, first, count in graph.split(self.group)
        }
        programs = [
            card.grid_program(
                plan, island_nocs, tensors, first_row=first, command=(reached[i], word)
            )
            for i, (plan, island_nocs, (first, _)) in enumerate(
                zip(self.plans, nocs, self.rows)
            )
        ]
        self.deadline = max(bench.deadline(plan) for plan in self.plans)
        self.run = session.load(
            programs,
            deadline=self.deadline,
            wait=wait,
        )

    def words(self, mode):
        addresses = [self.tensors[name].address for name in ("a", "w", "y")]
        return pl.grid_args(None, *addresses, mode)

    def calls(self, commands, mode=G["kGridFull"], timeout=60.0):
        """`commands` GRID commands back to back: (host seconds from the first submit
        to the last event, [commands, islands] microseconds of each part)."""
        words = np.tile(np.array(self.words(mode), np.int64), (commands, 1))
        started = time.perf_counter()
        handles = self.run.submit_many(
            "grid", self.group[0], self.group[1], words, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [[part.microseconds for part in event] for event in events]
        )

    def single(self, commands):
        """`commands` GRID commands one at a time (each waited for): the slowest
        island's microseconds each."""
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    def stop(self):
        """Stops the run; every island's kernel result and the cores that gave up."""
        self.run.stop()
        results = self.run.result()
        states = self.run.state()
        return {
            "cores": len(states),
            "kernel": [
                (result or {}).get("error_name")
                for result in (results if isinstance(results, list) else [results])
            ],
            "cores_ended": {
                key: sum(state[key] != 0 for state in states)
                for key in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {key: state[key] for key in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
            "broadcast_gave_up": [
                {"core": state["core"]} | area["gave_up"]
                for state in states
                for area in state["broadcasts"]
                if area["gave_up"]
            ][:8],
        }

    def failed(self, error):
        """After a failed wait: the run stopped if it can be, and the broadcasts'
        state (each read guarded)."""
        report = {"error": f"{type(error).__name__}: {error}"}
        try:
            self.run.stop()
        except km.KomorebiError as stop_error:
            report["stop"] = f"{type(stop_error).__name__}: {stop_error}"
        try:
            report["broadcasts"] = card.broadcasts_at_stall(self.run.state())
        except km.KomorebiError as state_error:
            report["state"] = f"{type(state_error).__name__}: {state_error}"
        return report


def clean(stopped):
    """No kernel error on any island, no core or broadcast given up, every core
    ended."""
    ended = all(count == stopped["cores"] for count in stopped["cores_ended"].values())
    gave_up = stopped["gave_up"] or stopped["broadcast_gave_up"]
    return (
        all(kernel == "none" for kernel in stopped["kernel"]) and ended and not gave_up
    )


def measure(product, cards, seconds, report):
    """Three single calls, then calls back to back for about `seconds` with every
    card's AICLK sampled."""
    single = product.single(3)
    one_microseconds = float(np.median(single))
    report["single_microseconds"] = [float(value) for value in single]
    log(f"single call {one_microseconds:.1f} us")
    reps = max(2, math.ceil(seconds * 1e6 / one_microseconds))
    with contextlib.ExitStack() as stack:
        clocks = [stack.enter_context(bench.Clock(index)) for index in cards]
        log(f"loop of {reps} calls")
        host_s, microseconds = product.calls(
            reps, timeout=3 * reps * one_microseconds / 1e6 + 5
        )
    report["loop"] = {
        "reps": reps,
        "seconds": round(host_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / reps, 3),
        "island_microseconds_median": np.median(microseconds, axis=0).tolist(),
        "aiclk_mhz": {
            card_index: clock.summary() for card_index, clock in zip(cards, clocks)
        },
    }
    log(f"loop: {report['loop']['microseconds_per_call']} us per call")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=run.SHAPES, default=None)
    parser.add_argument("--m", type=int, default=7680)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    dimensions = run.SHAPES[args.shape] if args.shape else (args.m, args.k, args.n)
    if any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    if args.inputs == "int" and dimensions[1] > run.EXACT_K:
        raise SystemExit(f"--inputs int needs K <= {run.EXACT_K}")
    cards = [int(value) for value in args.cards.split(",")]
    m_rows, k_depth, n_columns = dimensions
    report = {"shape": args.shape, "m": m_rows, "k": k_depth, "n": n_columns}
    report["cards"] = cards
    log("building the inputs")
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns, kind=args.inputs)
    log("inputs built")
    with km.Session(cards=cards, log=log) as session:
        started = time.perf_counter()
        y_matrix = np.zeros((m_rows, n_columns), np.float32)
        tensors = {}
        for name, value, placement in (
            ("a", a_matrix, {"by": "island"}),
            ("w", w_matrix, {"copy": True}),
            ("y", y_matrix, {"by": "island"}),
        ):
            tensors[name] = session.tensor(value, "bfloat16", "tile", **placement)
            log(f"{name.upper()} uploaded")
        report["upload_s"] = round(time.perf_counter() - started, 3)
        log(f"operands uploaded in {report['upload_s']} s")
        product = Rows(session, dimensions, tensors)
        report["islands"] = [
            {"rows": list(rows), "plan": run.describe(plan)}
            for rows, plan in zip(product.rows, product.plans)
        ]
        report["group"] = list(product.group)
        report["x280_mhz"] = [one.pll_mhz for one in session.sessions]
        log(f"loaded on {len(product.plans)} islands, group {product.group}")
        try:
            measure(product, cards, args.seconds, report)
        except km.KomorebiError as error:
            report["failed"] = product.failed(error)
            log(f"failed: {json.dumps(report['failed'], default=str)}")
            print(json.dumps(report, default=str), flush=True)
            sys.exit(1)
        report["run"] = product.stop()
        started = time.perf_counter()
        rows = ref.sampled_rows(m_rows)
        got = card.read_rows(session, tensors["y"], rows, log)
        report["readback_s"] = round(time.perf_counter() - started, 3)
        log("Y's sampled rows read back")
    flops = 2 * m_rows * k_depth * n_columns
    report["gflops"] = round(flops / report["loop"]["microseconds_per_call"] / 1e3, 1)
    report["sampled_row_tiles"] = sorted({int(row) // pl.TILE for row in rows})
    report["check"] = run.check(
        got[:, :n_columns], rows, a_matrix, w_matrix, args.inputs
    )
    report["ok"] = report["check"]["ok"] and clean(report["run"])
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
