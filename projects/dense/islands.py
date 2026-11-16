"""The matrix product over the islands of a session: A and Y split along their row
tiles (`split=0`), W a copy on every island; each island runs the grid plan of its own
rows, one GRID command an island a call on its rect (one card: every row).
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
from komorebi.bench import power  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

G = pl.G
log = run.log


def w_placement(session, far):
    """B's placement: a copy on every island, or (`far`) held by island 0 alone (split
    over its ranks), its pages read across by island 1's cores."""
    return {"split": 0, "group": (0, session.graph.count(0))} if far else {"copy": True}


def island_rows(a_tensor):
    """[(first row tile, row tiles)] of each island: its part of A's row tiles."""
    return list(a_tensor.placement.parts)


class Rows:
    """The product loaded on every island: island i's grid plan on its rows of A and
    Y, one GRID command an island a call on its rect's flat ranks, exclusive (its
    broadcasts)."""

    def __init__(self, session, dimensions, tensors, wait=1.0):
        graph, sessions = session.graph, session.sessions
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.tensors = tensors
        self.rows = island_rows(tensors["a"])
        if any(rows == 0 for _, rows in self.rows):
            raise SystemExit(f"{m_tiles} row tiles leave an island without rows")
        nocs = [one.nocs() for one in sessions]
        self.plans = [
            pl.grid_plan(rows, k_tiles, n_tiles, grid=island_nocs.grid)
            for (_, rows), island_nocs in zip(self.rows, nocs)
        ]
        if None in self.plans:
            raise SystemExit(f"no grid plan fits L1 for {dimensions}")
        # one command an island a call, on its rect's ranks: a command whose group
        # reaches another island crosses, and a crossing command is never exclusive
        rects = [plan.group(n.grid) for plan, n in zip(self.plans, nocs)]
        self.groups = [
            (graph.first(i) + first, count) for i, (first, count) in enumerate(rects)
        ]
        programs = [
            card.grid_program(
                plan,
                island_nocs,
                tensors,
                first_row=first,
                command=(rects[i], self.groups[i][0] | self.groups[i][1] << 16),
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
        to the last event, [commands, islands] microseconds of each island's)."""
        islands = len(self.groups)
        words = np.tile(np.array(self.words(mode), np.int64), (commands * islands, 1))
        firsts, counts = np.tile(np.array(self.groups, np.int64).T, commands)
        started = time.perf_counter()
        handles = self.run.submit_many("grid", firsts, counts, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [part.microseconds for event in events for part in event]
        ).reshape(commands, islands)

    def single(self, commands):
        """`commands` GRID commands one at a time (each waited for): the slowest
        island's microseconds each."""
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    def cross(self, session, timeout=10.0):
        """B's pages, island 0's copy, sent by its link cores into the same pages of
        island 1's copy and waited for there (two islands): host seconds."""
        w = self.tensors["w"]
        started = time.perf_counter()
        far = (1, session.wait_pages(1, session.link_cores[1][0], w.pages))
        session.wait_together([session.send_pages(w, 0, w.pages, w), far], timeout)
        return time.perf_counter() - started

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


def measure_moved(product, session, cards, seconds, report):
    """Each call B crosses from island 0's copy into island 1's, then one GRID
    command, waited for; for about `seconds`, every card's AICLK sampled."""
    first_cross = product.cross(session)
    single = product.single(3)
    one_microseconds = float(np.median(single)) + first_cross * 1e6
    report["single_microseconds"] = [float(value) for value in single]
    log(f"first crossing {first_cross * 1e3:.2f} ms, single call {single[0]:.1f} us")
    reps = max(2, math.ceil(seconds * 1e6 / one_microseconds))
    crossing_s, product_s, microseconds = 0.0, 0.0, []
    with contextlib.ExitStack() as stack:
        clocks = [stack.enter_context(bench.Clock(index)) for index in cards]
        log(f"loop of {reps} calls, each with B's crossing")
        started = time.perf_counter()
        for _ in range(reps):
            crossing_s += product.cross(session)
            host_s, parts = product.calls(1, timeout=10.0)
            product_s += host_s
            microseconds.append(parts[0])
        wall_s = time.perf_counter() - started
    pages = product.tensors["w"].pages
    report["loop"] = {
        "reps": reps,
        "seconds": round(wall_s, 3),
        "microseconds_per_call": round(wall_s * 1e6 / reps, 3),
        "crossing_microseconds_per_call": round(crossing_s * 1e6 / reps, 3),
        "product_microseconds_per_call": round(product_s * 1e6 / reps, 3),
        "crossing_gb_per_s": round(
            pages * product.tensors["w"].aligned_page_bytes * reps / crossing_s / 1e9, 2
        ),
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
    parser.add_argument("--links", type=int, default=0, help="page links, two cards")
    parser.add_argument("--kernel-bytes", type=int, default=None, help="kernel buffer")
    parser.add_argument(
        "--b-moved", action="store_true", help="B moved into island 1 each call"
    )
    parser.add_argument(
        "--b-far",
        action="store_true",
        help="B on island 0 only, read by island 1's cores across the link",
    )
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
    report |= {"cards": cards, "links": args.links, "b_moved": args.b_moved}
    report["b_far"] = args.b_far
    log("building the inputs")
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns, kind=args.inputs)
    log("inputs built")
    options = {"links": args.links, "kernel_bytes": args.kernel_bytes}
    with km.Session(cards=cards, log=log, **options) as session:
        started = time.perf_counter()
        y_matrix = np.zeros((m_rows, n_columns), np.float32)
        tensors = {}
        for name, value, placement in (
            ("a", a_matrix, {"split": 0}),
            ("w", w_matrix, w_placement(session, args.b_far)),
            ("y", y_matrix, {"split": 0}),
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
        report["groups"] = [list(group) for group in product.groups]
        report["x280_mhz"] = [one.pll_mhz for one in session.sessions]
        log(f"loaded on {len(product.plans)} islands, groups {product.groups}")
        try:
            if args.b_moved:
                measure_moved(product, session, cards, args.seconds, report)
            else:
                measure(product, cards, args.seconds, report)
        except km.KomorebiError as error:
            report["failed"] = product.failed(error)
            log(f"failed: {json.dumps(report['failed'], default=str)}")
            print(json.dumps(power.clocked(report), default=str), flush=True)
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
    print(json.dumps(power.clocked(report), default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
