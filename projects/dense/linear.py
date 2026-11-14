"""A linear layer Y = X x W written for islands (kernels/linear): one command a call on
every rank; X and W split along K alike, Y along N, by hints. Each island puts the
columns others keep and adds what they put. Prints one JSON line; exit 1: a failed
check."""

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
from komorebi import abi  # noqa: E402
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
# The kernel buffer the linear kernels' code takes (all five CPUs): one card 83,008 B;
# two cards (a rank's kernels: the page calls' client side) 94,224 B.
KERNEL_BYTES = {"one": 88 * 1024, "two": 96 * 1024}
log = run.log


def even_group(graph):
    """The hint's group that gives each island half of a split dimension: the last
    rank of island 0 and the first of island 1 (None on one island)."""
    islands = len(graph.islands)
    if islands == 1:
        return None
    if islands != 2:
        raise SystemExit(f"the halves' group is for two islands, not {islands}")
    return (graph.first(1) - 1, 2)


def spans(placement, outer_split):
    """Each island's (first, count) of the split dimension `outer_split` of a [rows,
    columns] tile tensor, as the kernels' k_part() and columns() find it."""
    out = []
    for first, steps in placement.parts:
        if placement.copy or placement.pages == steps * placement.inner * (
            placement.outer
        ):
            out.append((0, placement.steps))
        elif steps and placement.split == outer_split:
            out.append((first, steps))
        else:
            out.append((0, 0))
    return out


def sent_pages(plan, islands):
    """S's pages on an island: a set each other island (one at least), the output
    blocks of every plan core."""
    cores = pl.cdiv(plan.mt, plan.pcm) * pl.cdiv(plan.nt, plan.pcn)
    return max(1, islands - 1) * cores * plan.nbh * plan.nbw * plan.bm * plan.bn


def extra_buffers(plan, islands):
    """The linear kernels' buffers past the grid plan's: an output block to put across
    (a subblock on one island, where nothing is put), another island's subblock put
    here, the token and its slot."""
    subblock = plan.sh * plan.sw
    send = plan.bm * plan.bn if islands > 1 else subblock
    return [
        (L["kCbSend"], "bfloat16", L["kBf16Bytes"], send),
        (L["kCbSent"], "bfloat16", L["kBf16Bytes"], 2 * subblock),
        (L["kCbToken"], "uint32", L["kSignalBytes"], 1),
        (L["kCbArrive"], "uint32", L["kSignalBytes"], 1),
        (L["kCbStamp"], "uint32", L["kStampBytes"], 1),
    ]


def extra_bytes(plan, islands):
    return sum(page * pages for _, _, page, pages in extra_buffers(plan, islands))


def l1_budget(session, islands):
    """L1 for the buffers: the project's budget on one island; on more, what the kernel
    buffer leaves under the session's own L1 buffers at the top and the runtime's."""
    if islands == 1:
        return pl.L1_BUDGET
    worker = km.session.worker_l1_size(session.kernel_bytes)
    return min(pl.L1_BUDGET, worker - L1_TOP_LINKED - L1_RUNTIME)


def fitting_plan(m_tiles, k_tiles, n_tiles, grid, ranks, budget, islands):
    """The grid plan's rule on the island's grid with the extra buffers inside
    `budget`, with fewer rows of cores until its rect lies inside the island's ranks
    (link cores take the last cores)."""
    columns, rows = grid
    for used_rows in range(rows, 0, -1):
        room = budget
        while room > 0:
            plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, (columns, used_rows), room)
            if plan is None:
                break
            over = plan.l1_bytes() + extra_bytes(plan, islands) - budget
            if over <= 0:
                if plan.group(grid)[1] <= ranks:
                    return plan
                break
            room -= over
    return None


class Linear:
    """The linear kernels loaded on every island of `session`: one LINEAR command a
    call on all its ranks, exclusive (the broadcasts inside each island)."""

    def __init__(self, session, dimensions, tensors, wait=2.0, deadline=5.0):
        graph = session.graph
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.k_tiles = k_tiles
        counts = {graph.count(i) for i in range(len(graph.islands))}
        if len(counts) != 1:
            raise SystemExit(f"one program needs islands of one size: {counts}")
        ranks = counts.pop()
        self.islands = len(graph.islands)
        self.nocs = session.sessions[0].nocs()
        self.n_tiles = n_tiles
        self.k_parts = spans(tensors["w"].placement, 0)
        self.columns = spans(tensors["y"].placement, 1)
        if not all(count for _, count in self.k_parts):
            raise SystemExit(f"an island holds no K of W: {self.k_parts}")
        part_k = max(count for _, count in self.k_parts)
        part_n = max(count for _, count in self.columns)
        self.plan = fitting_plan(
            m_tiles,
            part_k,
            part_n,
            self.nocs.grid,
            ranks,
            l1_budget(session, self.islands),
            self.islands,
        )
        if self.plan is None:
            raise SystemExit(f"no grid plan fits L1 and {ranks} ranks: {dimensions}")
        for name in ("kCbRow", "kCbCol", "kCbPart"):
            assert L[name] == G[name], name
        assert L["kCbYOut"] == G["kCbGridOut"]
        pages = sent_pages(self.plan, self.islands)
        tensors["s"] = session.empty(
            (pages * pl.TILE, pl.TILE), "bfloat16", "tile", copy=True
        )
        tensors["t"] = session.empty(
            (ranks, L["kStampBytes"] // 4), "uint32", "row", copy=True
        )
        self.tensors = tensors
        self.session = session
        self.group = (0, session.ranks)
        word = self.group[0] | self.group[1] << 16
        program = card.grid_program(
            self.plan,
            self.nocs,
            {"a": tensors["x"], **tensors},
            command=((0, ranks), word),
        )
        x, w, y, s = (tensors[name] for name in "xwys")
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
                "reader": x.accessor_args() + s.accessor_args(),
                "writer": w.accessor_args()
                + y.accessor_args()
                + tensors["t"].accessor_args(),
                "compute": [],
            },
            compute={
                "math_fidelity": "HiFi2",
                "fp32_dest_acc_en": True,
                "unpack_to_dest_mode": {L["kCbPart"]: "UnpackToDestFp32"},
            },
            handlers=LINEAR_H,
        )
        self.run = session.load(self.program, deadline=deadline, wait=wait)

    def words(self):
        addresses = [self.tensors[name].address for name in "xwys"]
        return addresses + [self.k_tiles, self.n_tiles, self.tensors["t"].address]

    def passes(self, mhz=1350.0):
        """The last call's passes on each island from the writers' wall clocks (AICLK
        cycles, and microseconds at `mhz`): the puts sent, the sync, its own columns."""
        out = []
        for island, one in enumerate(self.session.sessions):
            (raw,) = one.read(self.tensors["t"].parts[island])
            words = (
                raw.view(np.uint32).reshape(-1, L["kStampBytes"] // 4).astype(np.int64)
            )
            ran = words[words[:, L["kStampEnd"]] != 0]
            stamped = ran[:, L["kStampPut"]].all()
            # the 32-bit clock as signed cycles from the first core's start
            ran = (ran - ran[0, L["kStampStart"]] + 2**31) % 2**32 - 2**31
            start = ran[:, L["kStampStart"]].min()
            put = ran[:, L["kStampPut"]].max() if stamped else start
            synced = ran[:, L["kStampSynced"]].max() if put != start else start
            end = ran[:, L["kStampEnd"]].max()
            cycles = {"put": put - start, "sync": synced - put, "own": end - synced}
            out.append(
                {
                    "cores": len(ran),
                    "cycles": {key: int(value) for key, value in cycles.items()},
                    "microseconds_at_mhz": {
                        key: round(float(value) / mhz, 1)
                        for key, value in cycles.items()
                    },
                }
            )
        return out

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
        """Bytes a call should cross, from the placements and the plan: each island
        holding part of K puts, for every other island keeping columns, each plan
        core's output blocks that hold some of them (whole blocks, bf16); and the
        ideal: every other holder's M x columns tiles."""
        plan = self.plan
        page = self.tensors["s"].aligned_page_bytes
        places = plan.places(self.nocs.grid).values()
        total = {"blocks": 0, "tiles": 0}
        for sender, (_, k_part) in enumerate(self.k_parts):
            for to, (_, width) in enumerate(self.columns):
                if to == sender or not k_part or not width:
                    continue
                total["tiles"] += plan.mt * width * page
                for first_row, first_column in places:
                    row_end = min(first_row + plan.pcm, plan.mt)
                    column_end = min(first_column + plan.pcn, width)
                    for block in range(plan.nbh * plan.nbw):
                        row = first_row + block // plan.nbw * plan.bm
                        column = first_column + block % plan.nbw * plan.bn
                        if row < row_end and column < column_end:
                            total["blocks"] += plan.bm * plan.bn * page
        total["put_bytes"] = plan.bm * plan.bn * page
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
    if isinstance(product, Linear):
        report["passes"] = product.passes()
        log(f"passes: {report['passes']}")
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
    parser.add_argument(
        "--kernel-bytes", type=int, default=None, help="kernel buffer (default: own)"
    )
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
    kernel_bytes = args.kernel_bytes or KERNEL_BYTES["two" if len(cards) > 1 else "one"]
    options = {"links": args.links, "kernel_bytes": kernel_bytes}
    options = {key: value for key, value in options.items() if value}
    with km.Session(cards=cards, log=log, **options) as session:
        started = time.perf_counter()
        m_rows, _, n_columns = dimensions
        # X and W split along K alike, Y along N, each island half (S, the tiles
        # other islands put, made with the kernels)
        halves = {"group": even_group(session.graph)}
        tensors = {
            "x": session.tensor(x_matrix, "bfloat16", "tile", split=1, **halves),
            "w": session.tensor(w_matrix, "bfloat16", "tile", split=0, **halves),
            "y": session.empty(
                (m_rows, n_columns), "bfloat16", "tile", split=1, **halves
            ),
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
