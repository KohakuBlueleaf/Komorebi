"""A linear layer Y = X x W written for islands: one program, one command a call on
every rank of the session; X, W and Y placed by the session's default (split along
the leading dimension), never a copy. Each island computes with the part of W it
holds (kernels/linear). Prints one JSON line; exits 1 when the check fails."""

import argparse
import contextlib
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi, layout  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

KERNELS = pl.HERE / "kernels" / "linear"
LINEAR_H = KERNELS / "linear.h"
L = abi.constants(LINEAR_H, known=abi.headers().constants)
G = pl.G
log = run.log


def held_rows(tensor, row_pages):
    """The most whole rows of `row_pages` pages any island holds of `tensor`."""
    return max(pages for _, pages in tensor.placement.parts) // row_pages


def extra_buffers(plan, others):
    """The linear kernels' buffers past the grid plan's: fp32 output subblocks, a
    share's chunks (this island's, each other island's), the token and its slot."""
    chunk = L["kChunk"]
    return [
        (L["kCbPartOut"], "float32", L["kPartBytes"], 2 * plan.sh * plan.sw),
        (L["kCbOwn"], "float32", L["kPartBytes"], 2 * chunk),
        (L["kCbFar"], "float32", L["kPartBytes"], 2 * chunk * max(others, 1)),
        (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
        (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
    ]


def extra_bytes(others):
    """L1 the extra buffers take at most (output subblocks of 4 tiles)."""
    chunk, page = L["kChunk"], L["kPartBytes"]
    return 8 * page + 2 * chunk * page * (1 + max(others, 1)) + 2 * L["kSignalBytes"]


def fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget):
    """The grid plan's rule on the island's grid, with fewer rows of cores until its
    rect lies inside the island's ranks (link cores take the last cores)."""
    columns, rows = grid
    for used_rows in range(rows, 0, -1):
        plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, (columns, used_rows), budget)
        if plan is not None and plan.group(grid)[1] <= ranks:
            return plan
    return None


def senders(plan, nocs, group):
    """The cores that read X: the row broadcast's senders (rank 0)."""
    row = card.grid_broadcasts(
        plan,
        nocs,
        group,
        "row",
        0,
        (G["kBroadcastRow"], G["kCbRow"], plan.wa, plan.bm * plan.kb),
    )
    places = plan.places(nocs.grid)
    return [core for core in places if row[core]["rank"] == 0]


class Linear:
    """The linear kernels loaded on every island of `session`: one LINEAR command a
    call on all its ranks, exclusive (the broadcasts inside each island)."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        graph = session.graph
        self.tensors = tensors
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.k_tiles = k_tiles
        counts = {graph.count(i) for i in range(len(graph.islands))}
        if len(counts) != 1:
            raise SystemExit(f"one program needs islands of one size: {counts}")
        ranks = counts.pop()
        self.islands = len(graph.islands)
        self.nocs = session.sessions[0].nocs()
        part_rows = held_rows(tensors["w"], n_tiles)
        budget = pl.L1_BUDGET - extra_bytes(self.islands - 1)
        self.plan = fitting_plan(
            m_tiles, part_rows, n_tiles, self.nocs.grid, ranks, budget
        )
        if self.plan is None:
            raise SystemExit(f"no grid plan fits L1 and {ranks} ranks: {dimensions}")
        for name in ("kCbRow", "kCbCol", "kCbPart"):
            assert L[name] == G[name], name
        assert L["kCbYOut"] == G["kCbGridOut"]
        self.group = (0, session.ranks)
        word = self.group[0] | self.group[1] << 16
        program = card.grid_program(
            self.plan,
            self.nocs,
            {"a": tensors["x"], **tensors},
            command=((0, ranks), word),
        )
        x, w, y, p = (tensors[name] for name in "xwyp")
        fp32 = "UnpackToDestFp32"
        self.program = dataclasses.replace(
            program,
            name="dense_linear",
            kernels=KERNELS,
            buffers=[
                km.Buffer(*buffer)
                for buffer in self.plan.buffers()[:4]
                + extra_buffers(self.plan, self.islands - 1)
            ],
            compile_args={
                "reader": x.accessor_args()
                + x.placement_args()
                + p.accessor_args()
                + y.placement_args()
                + w.placement_args(),
                "writer": w.accessor_args()
                + y.accessor_args()
                + y.placement_args()
                + p.accessor_args()
                + w.placement_args(),
                "compute": y.placement_args() + w.placement_args(),
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    L["kCbPart"]: fp32,
                    L["kCbOwn"]: fp32,
                    L["kCbFar"]: fp32,
                },
            },
            handlers=LINEAR_H,
        )
        self.x_senders = senders(self.plan, self.nocs, layout.Group(0, ranks))
        self.run = session.load(self.program, deadline=deadline, wait=wait)

    def words(self):
        return [self.tensors[name].address for name in "xwyp"] + [self.k_tiles]

    def calls(self, commands, timeout=60.0):
        """`commands` LINEAR commands back to back: (host seconds from the first
        submit to the last event, [commands, islands] microseconds of each part)."""
        words = np.tile(np.array(self.words(), np.int64), (commands, 1))
        started = time.perf_counter()
        handles = self.run.submit_many("linear", *self.group, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [part.microseconds for event in events for part in event]
        ).reshape(commands, -1)

    def single(self, commands):
        """`commands` calls one at a time: the slowest part's microseconds each."""
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    def crossed(self):
        """Bytes a call crosses, counted from the placements and the plan: X's rows
        another island holds, read by this island's X readers, and the partial sums
        of Y's pages each island asks of every other (none when W's part is all K)."""
        x, w, y = (self.tensors[name] for name in "xwy")
        plan, k_tiles = self.plan, self.k_tiles
        n_tiles = plan.nt
        places = plan.places(self.nocs.grid)
        total = {"x": 0, "partials": 0}
        for island, (w_first, w_pages) in enumerate(w.placement.parts):
            k_part = w_pages // n_tiles
            for core in self.x_senders:
                first_row = places[core][0]
                for block in range(plan.nbh * plan.nbw):
                    block_row = first_row + block // plan.nbw * plan.bm
                    for row in range(block_row, min(block_row + plan.bm, plan.mt)):
                        page = row * k_tiles + w_first // n_tiles
                        holders = x.holders(page, k_part)
                        total["x"] += (
                            sum(
                                pages
                                for holder, _, pages in holders
                                if holder != island
                            )
                            * x.aligned_page_bytes
                        )
            if k_part < k_tiles:
                total["partials"] += (
                    y.placement.parts[island][1]
                    * (self.islands - 1)
                    * self.tensors["p"].aligned_page_bytes
                )
        total["all"] = total["x"] + total["partials"]
        return total

    def stop(self):
        """Stops the run: each island's kernel result and the cores that gave up."""
        self.run.stop()
        results = self.run.result()
        states = self.run.state()
        return {
            "cores": len(states),
            "kernel": [
                (result or {}).get("error_name")
                for result in (results if isinstance(results, list) else [results])
            ],
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


class Grid:
    """The dense project's own grid kernels on a one-island session, beside: its
    plan for the whole product, one GRID command a call on its rect."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        nocs = session.sessions[0].nocs()
        self.plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, grid=nocs.grid)
        self.group = self.plan.group(nocs.grid)
        self.tensors = {"a": tensors["x"], "w": tensors["w"], "y": tensors["y"]}
        self.program = card.grid_program(self.plan, nocs, self.tensors)
        self.run = session.load(self.program, deadline=deadline, wait=wait)

    def calls(self, commands, timeout=60.0):
        addresses = [self.tensors[name].address for name in ("a", "w", "y")]
        words = np.tile(
            np.array(pl.grid_args(self.plan, *addresses, G["kGridFull"]), np.int64),
            (commands, 1),
        )
        started = time.perf_counter()
        handles = self.run.submit_many("grid", *self.group, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [part.microseconds for event in events for part in event]
        ).reshape(commands, -1)

    def single(self, commands):
        return np.concatenate(
            [self.calls(1, timeout=10.0)[1].max(axis=1) for _ in range(commands)]
        )

    stop = Linear.stop


def measure(product, cards, seconds):
    """Three single calls, then calls back to back for about `seconds` with every
    card's AICLK sampled."""
    single = product.single(3)
    one_microseconds = float(np.median(single))
    log(f"single call {one_microseconds:.1f} us")
    reps = max(2, math.ceil(seconds * 1e6 / one_microseconds))
    with contextlib.ExitStack() as stack:
        clocks = [stack.enter_context(bench.Clock(index)) for index in cards]
        log(f"loop of {reps} calls")
        host_s, microseconds = product.calls(
            reps, timeout=3 * reps * one_microseconds / 1e6 + 5
        )
    report = {
        "single_microseconds": [float(value) for value in single],
        "reps": reps,
        "seconds": round(host_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / reps, 3),
        "part_microseconds_median": np.median(microseconds, axis=0).tolist(),
        "aiclk_mhz": {
            card_index: clock.summary() for card_index, clock in zip(cards, clocks)
        },
    }
    log(f"loop: {report['microseconds_per_call']} us per call")
    return report


def clean(stopped):
    gave_up = stopped["gave_up"] or stopped["broadcast_gave_up"]
    return all(kernel == "none" for kernel in stopped["kernel"]) and not gave_up


def checked(session, tensors, dimensions, matrices, kind):
    """Y's sampled rows read back and checked against the float64 product."""
    m_rows, _, n_columns = dimensions
    rows = ref.sampled_rows(m_rows)
    started = time.perf_counter()
    got = card.read_rows(session, tensors["y"], rows, log)
    report = run.check(got[:, :n_columns], rows, *matrices, kind)
    report["readback_s"] = round(time.perf_counter() - started, 3)
    return report


def sizes(program):
    found = getattr(program, "sizes", None)
    return None if found is None else {"text": found.text, "total": found.total}


def attempt(product_class, session, dimensions, tensors, cards, args, matrices):
    """Loads, measures, stops and checks one product; its report."""
    flops = 2 * math.prod(dimensions)
    report = {}
    try:
        product = product_class(session, dimensions, tensors, wait=args.wait)
    except km.KomorebiError as error:
        return {"error": f"{type(error).__name__}: {error}", "ok": False}
    report["plan"] = run.describe(product.plan)
    report["kernel_sizes"] = sizes(product.program)
    if isinstance(product, Linear):
        report["crossed_bytes"] = product.crossed()
    log(f"{product_class.__name__} loaded, plan {report['plan']}")
    try:
        report["loop"] = measure(product, cards, args.seconds)
    except km.KomorebiError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        with contextlib.suppress(km.KomorebiError):
            product.run.stop()
        report["ok"] = False
        return report
    report["run"] = product.stop()
    report["check"] = checked(session, tensors, dimensions, matrices, args.inputs)
    per_call = report["loop"]["microseconds_per_call"]
    report["tflops"] = round(flops / per_call / 1e6, 2)
    if "crossed_bytes" in report and report["crossed_bytes"]["all"]:
        report["crossed_gb_per_s"] = round(
            report["crossed_bytes"]["all"] / per_call / 1e3, 2
        )
    report["ok"] = report["check"]["ok"] and clean(report["run"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=run.SHAPES, default=None)
    parser.add_argument("--cube", type=int, default=None, help="M = K = N")
    parser.add_argument("--cards", default="0")
    parser.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--wait", type=float, default=2.0, help="wait limit, s")
    parser.add_argument("--links", type=int, default=0, help="page links, two cards")
    parser.add_argument("--kernel-bytes", type=int, default=None, help="kernel buffer")
    parser.add_argument(
        "--beside", action="store_true", help="then the project's grid kernels"
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    dimensions = (args.cube,) * 3 if args.cube else run.SHAPES[args.shape]
    if any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    cards = [int(value) for value in args.cards.split(",")]
    report = {"m": dimensions[0], "k": dimensions[1], "n": dimensions[2]}
    report |= {"cards": cards, "links": args.links}
    log("building the inputs")
    x_matrix, w_matrix = ref.inputs(*dimensions, kind=args.inputs)
    log("inputs built")
    options = {"links": args.links, "kernel_bytes": args.kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    with km.Session(cards=cards, log=log, **options) as session:
        started = time.perf_counter()
        m_rows, _, n_columns = dimensions
        tensors = {
            "x": session.tensor(x_matrix, "bfloat16", "tile"),
            "w": session.tensor(w_matrix, "bfloat16", "tile"),
            "y": session.empty((m_rows, n_columns), "bfloat16", "tile"),
            "p": session.empty((m_rows, n_columns), "float32", "tile"),
        }
        report["upload_s"] = round(time.perf_counter() - started, 3)
        report["placements"] = {
            name: [list(part) for part in tensor.placement.parts]
            for name, tensor in tensors.items()
        }
        log(f"operands placed in {report['upload_s']} s: {report['placements']}")
        report["x280_mhz"] = [one.pll_mhz for one in session.sessions]
        matrices = (x_matrix, w_matrix)
        report["linear"] = attempt(
            Linear, session, dimensions, tensors, cards, args, matrices
        )
        if args.beside and report["linear"]["ok"]:
            log(f"cool-down {args.seconds} s")
            time.sleep(args.seconds)
            report["grid"] = attempt(
                Grid, session, dimensions, tensors, cards, args, matrices
            )
    report["ok"] = report["linear"]["ok"] and report.get("grid", {}).get("ok", True)
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
