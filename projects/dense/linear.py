"""A linear layer Y = X x W written for islands: one program, one command a call on
every rank of the session; X, W and Y placed by a hint (--split-w 0: W split along
K, 1: along N), never a copy. Each island computes with the part of W it holds
(kernels/linear). Prints one JSON line; exits 1 when the check fails."""

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
from komorebi import abi, formats, layout  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

KERNELS = pl.HERE / "kernels" / "linear"
LINEAR_H = KERNELS / "linear.h"
L = abi.constants(LINEAR_H, known=abi.headers().constants)
G = pl.G
# Two cards: L1 the session keeps at the top of a core (a clash at 1,474,944 of
# 1,555,968), and the runtime's control block, overlay area and reader slot.
L1_TOP_LINKED = 81_024
L1_RUNTIME = 16_960 + 16_256
log = run.log


def covers(w, k_tiles, n_tiles):
    """Each island's block of W's [K, N] tiles, (k0, K tiles, n0, N tiles), as the
    kernels' cover() finds it."""
    placement = w.placement
    out = []
    for first, steps in placement.parts:
        if placement.copy or placement.pages == steps * placement.inner:
            out.append((0, k_tiles, 0, n_tiles) if steps else (0, 0, 0, 0))
        elif placement.split == 0:
            out.append((first, steps, 0, n_tiles))
        else:
            out.append((0, k_tiles, first, steps))
    return out


def extra_buffers(plan, islands):
    """The linear kernels' buffers past the grid plan's: fp32 output subblocks, a
    share's chunk of every island's partial sums, the token and its slot."""
    chunk = L["kChunk"]
    return [
        (L["kCbPartOut"], "float32", L["kPartBytes"], 2 * plan.sh * plan.sw),
        (L["kCbSums"], "float32", L["kPartBytes"], 2 * chunk * islands),
        (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
        (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
    ]


def extra_bytes(islands):
    """L1 the extra buffers take at most (output subblocks of 4 tiles)."""
    chunk, page = L["kChunk"], L["kPartBytes"]
    return 8 * page + 2 * chunk * page * islands + 2 * L["kSignalBytes"]


def l1_budget(session, islands):
    """L1 for the buffers: the project's budget on one island; on more, what the kernel
    buffer leaves under the session's own L1 buffers at the top and the runtime's."""
    if islands == 1:
        return pl.L1_BUDGET
    worker = km.session.worker_l1_size(session.kernel_bytes)
    return min(pl.L1_BUDGET, worker - L1_TOP_LINKED - L1_RUNTIME)


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
        self.n_tiles = n_tiles
        self.covers = covers(tensors["w"], k_tiles, n_tiles)
        part_k = max(cover[1] for cover in self.covers)
        part_n = max(cover[3] for cover in self.covers)
        budget = l1_budget(session, self.islands) - extra_bytes(self.islands)
        self.plan = fitting_plan(m_tiles, part_k, part_n, self.nocs.grid, ranks, budget)
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
                + extra_buffers(self.plan, self.islands)
            ],
            compile_args={
                "reader": x.accessor_args() + p.accessor_args(),
                "writer": w.accessor_args() + y.accessor_args() + p.accessor_args(),
                "compute": [],
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {
                    L["kCbPart"]: fp32,
                    L["kCbSums"]: fp32,
                },
            },
            handlers=LINEAR_H,
        )
        self.x_senders = senders(self.plan, self.nocs, layout.Group(0, ranks))
        self.run = session.load(self.program, deadline=deadline, wait=wait)

    def words(self):
        addresses = [self.tensors[name].address for name in "xwyp"]
        return addresses + [self.k_tiles, self.n_tiles]

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
        """Bytes a call should cross, counted from the placements and the plan: X's rows
        another island holds, read by the X readers of an island holding part of W;
        Y's pages written across; and (no island holding all of K) every other
        holder's partial sums of the Y pages an island holds, asked."""
        x, y, p = (self.tensors[name] for name in "xyp")
        plan, k_tiles, n_tiles = self.plan, self.k_tiles, self.n_tiles
        places = plan.places(self.nocs.grid)
        holders = [island for island, cover in enumerate(self.covers) if cover[1]]
        whole = all(self.covers[island][1] == k_tiles for island in holders)
        total = {"x": 0, "y": 0, "partials": 0}
        for island, (k0, k_part, n0, n_part) in enumerate(self.covers):
            for core in self.x_senders if k_part else ():
                first_row = places[core][0]
                for block in range(plan.nbh * plan.nbw):
                    block_row = first_row + block // plan.nbw * plan.bm
                    for row in range(block_row, min(block_row + plan.bm, plan.mt)):
                        holders_x = x.holders(row * k_tiles + k0, k_part)
                        total["x"] += (
                            sum(pages for at, _, pages, _ in holders_x if at != island)
                            * x.aligned_page_bytes
                        )
            if whole and n_part:
                for row in range(plan.mt):
                    runs = y.holders(row * n_tiles + n0, n_part)
                    total["y"] += (
                        sum(pages for at, _, pages, _ in runs if at != island)
                        * y.aligned_page_bytes
                    )
            if not whole:
                others = len([at for at in holders if at != island])
                total["partials"] += (
                    y.placement.part_pages(island) * others * p.aligned_page_bytes
                )
        total["all"] = total["x"] + total["y"] + total["partials"]
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
    before = product.run.crossings()
    single = product.single(3)
    one_microseconds = float(np.median(single))
    log(f"single call {one_microseconds:.1f} us")
    if seconds <= 0:  # the single calls alone, each waited for
        crossed = (product.run.crossings() - before).total()
        return {
            "single_microseconds": [float(value) for value in single],
            "reps": 0,
            "microseconds_per_call": round(one_microseconds, 3),
            "crossed_a_call": {field: count / 3 for field, count in crossed.items()},
            "aiclk_mhz": {},
        }
    reps = max(2, math.ceil(seconds * 1e6 / one_microseconds))
    before = product.run.crossings()
    with contextlib.ExitStack() as stack:
        clocks = [stack.enter_context(bench.Clock(index)) for index in cards]
        log(f"loop of {reps} calls")
        host_s, microseconds = product.calls(
            reps, timeout=3 * reps * max(one_microseconds, 50.0) / 1e6 + 10
        )
    crossed = (product.run.crossings() - before).total()
    report = {
        "single_microseconds": [float(value) for value in single],
        "reps": reps,
        "seconds": round(host_s, 3),
        "microseconds_per_call": round(host_s * 1e6 / reps, 3),
        "part_microseconds_median": np.median(microseconds, axis=0).tolist(),
        "crossed_a_call": {field: count / reps for field, count in crossed.items()},
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
    """Y's sampled rows read back and checked against the float64 product; when the
    check fails on several islands, each sampled row tile's error and its holder."""
    m_rows, _, n_columns = dimensions
    rows = ref.sampled_rows(m_rows)
    started = time.perf_counter()
    got = card.read_rows(session, tensors["y"], rows, log)
    report = run.check(got[:, :n_columns], rows, *matrices, kind)
    report["readback_s"] = round(time.perf_counter() - started, 3)
    if not report["ok"] and len(session.sessions) > 1:
        want = ref.matmul(matrices[0][rows], matrices[1])
        y = tensors["y"]
        report["row_tiles"] = [
            {
                "row_tile": int(rows[at]) // pl.TILE,
                "holder": y.holders(
                    int(rows[at]) // pl.TILE * y.shape[1] // pl.TILE, 1
                )[0][0],
                **ref.errors(
                    got[at : at + pl.TILE, :n_columns], want[at : at + pl.TILE]
                ),
            }
            for at in range(0, len(rows), pl.TILE)
        ]
        report["partials"] = partials(session, tensors, matrices)
    return report


def partials(session, tensors, matrices):
    """Each island's partial sums in P (its last call) for the first row tile of
    each island's part of Y, against the float64 partial of its part of K."""
    x_matrix, w_matrix = matrices
    p, w, y = tensors["p"], tensors["w"], tensors["y"]
    n_tiles = p.shape[1] // pl.TILE
    out = []
    for island, cover in enumerate(covers(w, w.shape[0] // pl.TILE, n_tiles)):
        k0, k_part = cover[0] * pl.TILE, cover[1] * pl.TILE
        if not k_part:
            continue
        for at in range(len(y.placement.parts)):
            if not y.placement.part_pages(at):
                continue
            row_tile = int(y.placement.part(at)[0]) // n_tiles
            (raw,) = session.sessions[island].read(
                p.parts[island], first=row_tile * n_tiles, pages=n_tiles
            )
            got = formats.untilize(raw.view(np.float32), pl.TILE, n_tiles * pl.TILE)
            rows = slice(row_tile * pl.TILE, (row_tile + 1) * pl.TILE)
            want = ref.matmul(
                x_matrix[rows, k0 : k0 + k_part], w_matrix[k0 : k0 + k_part]
            )
            out.append(
                {"island": island, "row_tile": row_tile, **ref.errors(got, want)}
            )
    return out


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
    crossed = report["loop"]["crossed_a_call"]["bytes"]
    if crossed:
        report["crossed_gb_per_s"] = round(crossed / per_call / 1e3, 2)
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
    parser.add_argument("--split-w", type=int, choices=[0, 1], default=0)
    parser.add_argument("--split-x", type=int, choices=[0, 1], default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    dimensions = (args.cube,) * 3 if args.cube else run.SHAPES[args.shape]
    if any(value % pl.TILE for value in dimensions):
        raise SystemExit(f"M, K and N are whole tiles of {pl.TILE}: {dimensions}")
    cards = [int(value) for value in args.cards.split(",")]
    report = {"m": dimensions[0], "k": dimensions[1], "n": dimensions[2]}
    report |= {"cards": cards, "links": args.links}
    report |= {"split_w": args.split_w, "split_x": args.split_x}
    log("building the inputs")
    x_matrix, w_matrix = ref.inputs(*dimensions, kind=args.inputs)
    log("inputs built")
    options = {"links": args.links, "kernel_bytes": args.kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    with km.Session(cards=cards, log=log, **options) as session:
        started = time.perf_counter()
        m_rows, _, n_columns = dimensions
        # Y split as W's N is (rows when W is split along K); P, an island's
        # partial sums of all of Y, is scratch on every island
        tensors = {
            "x": session.tensor(x_matrix, "bfloat16", "tile", split=args.split_x),
            "w": session.tensor(w_matrix, "bfloat16", "tile", split=args.split_w),
            "y": session.empty(
                (m_rows, n_columns), "bfloat16", "tile", split=args.split_w
            ),
            "p": session.empty((m_rows, n_columns), "float32", "tile", copy=True),
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
