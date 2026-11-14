"""The matrix product's grid form, kernels unchanged, on ONE flat group over the islands
of a session: the project's rule makes one plan for the islands' grids stood one below
the other, and its broadcasts span the islands where its columns do. Each tensor is
held once. Prints one JSON line; exits 1 when the check fails. Usage in README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import channel, layout  # noqa: E402
from projects.dense import bench, card, islands, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

G = pl.G
log = run.log
STACK = "below"  # rows stay in an island: A and Y split by rows never cross


class Flat:
    """The flat plan: `plan` (the rule's, on the islands' grids one below the other,
    each less the column of its link cores) at row `y0` of that whole grid, so that
    island 0's bands end where its part of A's row tiles (`split`) ends."""

    def __init__(self, graph, nocs, dimensions, split):
        self.graph, self.nocs = graph, nocs
        m_tiles, k_tiles, n_tiles = (value // pl.TILE for value in dimensions)
        self.columns, self.rows = nocs[0].grid
        self.islands = len(nocs)
        usable = self.columns - (1 if self.islands > 1 else 0)  # link core 0: last
        self.grid = (usable, self.rows * self.islands)
        self.plan = pl.grid_plan(m_tiles, k_tiles, n_tiles, grid=self.grid)
        if self.plan is None:
            raise SystemExit(f"no grid plan fits L1 for {dimensions}")
        self.y0, self.aligned = 0, False
        plan = self.plan
        if self.islands > 1 and plan.layout == "2d" and split % plan.pcm == 0:
            above = split // plan.pcm  # bands on island 0
            if above <= self.rows and plan.rows - above <= self.rows:
                self.y0, self.aligned = self.rows - above, True
        x0, y0, width, height = plan.rect(self.grid)
        self.rect = (x0, y0 + self.y0, width, height)
        self.places = {}  # flat rank: (first row tile, first column tile)
        for core, place in plan.places(self.grid).items():
            y, x = divmod(core, self.grid[0])
            self.places[self.rank(x, y + self.y0)] = place
        ranks = sorted(self.places)
        self.group = layout.Group(ranks[0], ranks[-1] - ranks[0] + 1)
        self.touched = sorted({y // self.rows for y in range(y0, y0 + height)})

    def hints(self):
        """Split hints under which no page crosses: every tensor over the ranks of
        the one island the rect lies on; or A and Y split along their rows over the
        two ranks at the islands' edge (half an island, where the bands meet) and W
        over the bottom island, whose cores send the columns' blocks. None else."""
        graph = self.graph
        if len(self.touched) == 1:
            island = self.touched[0]
            ranks = {"group": (graph.first(island), graph.count(island))}
            return {"a": ranks, "w": ranks, "y": ranks}
        if not self.aligned:
            return None
        last = self.islands - 1
        edge = {"split": 0, "group": (graph.first(1) - 1, 2)}
        bottom = {"group": (graph.first(last), graph.count(last))}
        return {"a": edge, "w": bottom, "y": edge}

    def rank(self, x, y):
        """The flat rank of cell (x, y) of the whole grid."""
        island, local = divmod(y, self.rows)
        core = local * self.columns + x
        if core >= self.graph.count(island):
            raise SystemExit(f"cell {(x, y)} is island {island}'s link core")
        return self.graph.first(island) + core

    def broadcasts(self, along, noc, buffer):
        """Each island's broadcast words by core (layout.island_broadcast_args), a
        broadcast of one core each where the layout gives every core its operand."""
        islands = [
            (one, self.graph.first(i), self.graph.count(i))
            for i, one in enumerate(self.nocs)
        ]
        common = (islands, self.group)
        out = layout.island_broadcast_args(
            *common, self.rect, along, noc, buffer, STACK
        )
        if {"row": "rows", "column": "cols"}[along] != self.plan.layout:
            return out
        x0, y0, width, height = self.rect
        for y in range(y0, y0 + height):
            for x in range(x0, x0 + width):
                single = layout.island_broadcast_args(
                    *common, (x, y, 1, 1), along, noc, buffer, STACK
                )
                island, local = divmod(y, self.rows)
                core = local * self.columns + x
                out[island][core] = single[island][core]
        return out

    def program_args(self):
        """Each island's args(core) of the grid kernels."""
        plan = self.plan
        row = self.broadcasts(
            "row", 0, (G["kBroadcastRow"], G["kCbRow"], plan.wa, plan.bm * plan.kb)
        )
        column = self.broadcasts(
            "column", 1, (G["kBroadcastCol"], G["kCbCol"], plan.ww, plan.kb * plan.bn)
        )
        idle = [0] * (len(card.BROADCAST_FIELDS) + G["kCoreWords"])
        words = plan.words()

        def island_args(island):
            first = self.graph.first(island)

            def args(core):
                rank = first + core
                if core >= self.graph.count(island) or rank not in self.group.indices:
                    return idle, idle, idle[: G["kCoreWords"]] + [plan.wa, plan.ww]
                row_tile, column_tile = self.places.get(rank, (0, 0))
                mine = [int(rank in self.places), row_tile, column_tile, *words]
                broadcasts = [
                    [out[island][core][field] for field in card.BROADCAST_FIELDS]
                    for out in (row, column)
                ]
                return (
                    broadcasts[0] + mine,
                    broadcasts[1] + mine,
                    mine
                    + [
                        plan.wa,
                        plan.ww,
                    ],
                )

            return args

        return [island_args(island) for island in range(self.islands)]


def flat_program(plan, args, tensors, fidelity="HiFi2"):
    """card.grid_program's program with these args."""
    return km.Program(
        "dense_grid",
        kernels=card.GRID_KERNELS,
        buffers=[km.Buffer(*buffer) for buffer in plan.buffers()],
        args=args,
        compile_args={
            "reader": tensors["a"].tensor_args(),
            "writer": tensors["w"].tensor_args() + tensors["y"].tensor_args(),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {G["kCbPart"]: "UnpackToDestFp32"},
        },
        handlers=pl.GRID_H,
    )


class FlatProduct(islands.Rows):
    """The flat plan loaded on every island; one GRID command a call on the flat group,
    split by the host into a part an island, exclusive (its broadcasts)."""

    def __init__(self, session, flat, tensors, wait=1.0):
        self.tensors, self.flat = tensors, flat
        self.groups = [(flat.group.first, flat.group.count)]
        programs = [
            flat_program(flat.plan, args, tensors) for args in flat.program_args()
        ]
        self.deadline = bench.deadline(flat.plan)
        if len(flat.touched) > 1:
            # pages and blocks crossing at a few GB/s: three times the call at 2 TFLOP/s
            self.deadline = round(3 * pl.flops(flat.plan) / 2e12 + 0.05, 3)
        self.run = session.load(programs, deadline=self.deadline, wait=wait)

    def calls(self, commands, mode=G["kGridFull"], timeout=60.0):
        """`commands` GRID commands back to back: (host seconds from the first submit
        to the last event, [commands, parts] microseconds of each island's part)."""
        words = np.tile(np.array(self.words(mode), np.int64), (commands, 1))
        first, count = self.groups[0]
        started = time.perf_counter()
        handles = self.run.submit_many("grid", first, count, words, exclusive=True)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [part for event in events for part in event if part.kind != "done"]
        if unfinished:
            raise km.KomorebiError(f"{len(unfinished)} parts not done: {unfinished[0]}")
        return host_s, np.array(
            [[part.microseconds for part in event] for event in events]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=run.SHAPES, default=None)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--cards", default="0,1")
    parser.add_argument("--inputs", choices=["uniform", "int"], default="uniform")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--hints",
        action="store_true",
        help="split hints that keep pages home (else none: pages may cross)",
    )
    parser.add_argument(
        "--links", type=int, default=1, help="links between two cards (1 to 4)"
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
    report |= {"cards": cards, "stack": STACK}
    log("building the inputs")
    a_matrix, w_matrix = ref.inputs(m_rows, k_depth, n_columns, kind=args.inputs)
    options = {"links": args.links} if len(cards) > 1 else {}
    with km.Session(cards=cards, log=log, **options) as session:
        graph = session.graph
        nocs = [one.nocs() for one in session.sessions]
        m_tiles = m_rows // pl.TILE
        split = channel.partition(m_tiles, 0, len(nocs))[1] if len(nocs) > 1 else 0
        flat = Flat(graph, nocs, dimensions, split)
        placements = flat.hints() if args.hints else {"a": {}, "w": {}, "y": {}}
        if placements is None:
            raise SystemExit(f"{dimensions}: no hints keep A's and Y's pages home")
        started = time.perf_counter()
        y_matrix = np.zeros((m_rows, n_columns), np.float32)
        tensors = {}
        for name, value in (("a", a_matrix), ("w", w_matrix), ("y", y_matrix)):
            tensors[name] = session.tensor(
                value, "bfloat16", "tile", **placements[name]
            )
        report["upload_s"] = round(time.perf_counter() - started, 3)
        report["hints"] = placements
        report["a_rows_by_island"] = [
            list(part) for part in tensors["a"].placement.parts
        ]
        report["plan"] = run.describe(flat.plan) | {"rect": list(flat.rect)}
        report["group"] = [flat.group.first, flat.group.count]
        log(f"plan {report['plan']}, group {report['group']}")
        product = FlatProduct(session, flat, tensors)
        log("loaded")
        before = product.run.crossings()
        try:
            islands.measure(product, cards, args.seconds, report)
        except km.KomorebiError as error:
            report["failed"] = product.failed(error)
            log(f"failed: {json.dumps(report['failed'], default=str)}")
            print(json.dumps(report, default=str), flush=True)
            sys.exit(1)
        crossed = product.run.crossings() - before
        calls = 3 + report["loop"]["reps"]
        report["crossed_a_call"] = {
            key: round(value / calls, 1) for key, value in crossed.total().items()
        }
        report["parts_by_link"] = {  # the relay's parts a call, link and direction
            f"{link} {source}->{target}": round(row["parts"] / calls, 1)
            for (link, source, target), row in sorted(crossed.rows.items())
        }
        report["run"] = product.stop()
        rows = ref.sampled_rows(m_rows)
        got = card.read_rows(session, tensors["y"], rows, log)
    flops = 2 * m_rows * k_depth * n_columns
    report["tflops"] = round(flops / report["loop"]["microseconds_per_call"] / 1e6, 2)
    report["sampled_row_tiles"] = sorted({int(row) // pl.TILE for row in rows})
    report["check"] = run.check(
        got[:, :n_columns], rows, a_matrix, w_matrix, args.inputs
    )
    report["ok"] = report["check"]["ok"] and islands.clean(report["run"])
    report["load_avg"] = list(Path("/proc/loadavg").read_text().split()[:3])
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
