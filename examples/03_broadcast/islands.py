"""Broadcast over two cards (`Session(cards=..., links=1)`), its kernels unchanged: one
flat group over both islands, a rectangle whose columns ("below") or rows ("beside")
span the islands; each core's checksum against numpy. Also one card, or one split into
islands of rows (`--islands`, `--line`). Prints one JSON line. See README.md."""

import argparse
import json
import sys
import time

import numpy as np
from broadcast import BROADCAST_FIELDS, COLUMN, CONSTANTS, KERNELS, ROW, TILE

import komorebi as km
from komorebi import layout

STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--islands", type=int, default=0, help="one card split into this many islands"
    )
    parser.add_argument(
        "--line",
        action="store_true",
        help="--islands joined in a line (neighbours only), not each to each",
    )
    parser.add_argument("--blocks", type=int, default=8, help="blocks n a broadcast")
    parser.add_argument("--pages", type=int, default=2, help="tiles P per block")
    parser.add_argument("--window", type=int, default=2, help="blocks B in flight")
    parser.add_argument("--jobs", type=int, default=8, help="CAST commands a case")
    parser.add_argument("--wait", type=float, default=2.0, help="wait limit, seconds")
    parser.add_argument(
        "--deadline", type=float, default=1.0, help="a command, seconds"
    )
    return parser.parse_args()


def bands(session):
    """A split card: each island's (first row, rows) of the card's grid; else None."""
    if len(session.sessions) == 1 or session.sessions[0] is not session.sessions[1]:
        return None
    columns = session.island(0).nocs().grid[0]
    return [
        (island.cores[0] // columns, len(island.cores) // columns)
        for island in session.graph.islands
    ]


def cases(session, grid):
    """(stack, rect): 6 x 4 cells across the islands' edge, columns spanning when the
    grids stand one below the other, rows when side by side; one card: the 6 x 4 at
    the bottom of its grid; a split card: 6 columns down every row, through every
    island."""
    columns, rows = grid
    if bands(session) is not None:
        return [("below", (0, 0, 6, rows))]
    if len(session.sessions) == 1:
        return [("below", (0, rows - 4, 6, 4))]
    return [("below", (0, rows - 2, 6, 4)), ("beside", (columns - 3, 0, 6, 4))]


def plan(session, stack, rect, args):
    """Each island's program arguments (reader, writer, compute) by core, and the
    rectangle's cells by flat rank: (row, column)."""
    graph = session.graph
    islands = [
        (session.island(i).nocs(), graph.first(i), graph.count(i))
        for i in range(len(session.sessions))
    ]
    columns, rows = islands[0][0].grid
    split = bands(session)
    group = layout.Group(0, session.ranks)
    window, pages = args.window, args.pages
    by_broadcast = {
        COLUMN: layout.island_broadcast_args(
            islands,
            group,
            rect,
            "column",
            1,
            (COLUMN, CONSTANTS["kCbCol"], window, pages),
            stack,
            split,
        ),
        ROW: layout.island_broadcast_args(
            islands,
            group,
            rect,
            "row",
            0,
            (ROW, CONSTANTS["kCbRow"], window, pages),
            stack,
            split,
        ),
    }
    x0, y0, width, height = rect
    cells = {}
    for y in range(y0, y0 + height):
        for x in range(x0, x0 + width):
            island = y // rows if stack == "below" else x // columns
            core = (y % rows) * columns + x % columns
            if split is not None:  # the card's rows, an island's band
                (island,) = [i for i, (top, n) in enumerate(split) if 0 <= y - top < n]
                core = (y - split[island][0]) * columns + x
            cells[graph.first(island) + core] = (y - y0, x - x0)

    def program_args(island):
        def core_args(core):
            rank = graph.first(island) + core
            if core >= graph.count(island):  # a link core: not a rank
                idle = [0] * (len(BROADCAST_FIELDS) + 1)
                return idle, idle, [0, window, pages]
            row, column = cells.get(rank, (0, 0))
            words = {
                broadcast: [
                    by_broadcast[broadcast][island][core][field]
                    for field in BROADCAST_FIELDS
                ]
                for broadcast in by_broadcast
            }
            included = int(rank in cells)
            return (
                words[ROW] + [row],
                words[COLUMN] + [column],
                [included, window, pages],
            )

        return core_args

    if split is not None:  # one program: a core of the card, its island's arguments
        starts = [top * columns for top, _ in split]

        def card_args(core):
            island = max(i for i, start in enumerate(starts) if start <= core)
            return program_args(island)(core - starts[island])

        return group, [card_args], cells
    return group, [program_args(i) for i in range(len(islands))], cells


def program(core_args, row_tensor, column_tensor, output, args):
    window, pages = args.window, args.pages
    return km.Program(
        "broadcast",
        kernels=KERNELS,
        buffers=[
            km.Buffer(
                CONSTANTS["kCbRow"], "bfloat16", CONSTANTS["kTileBytes"], window * pages
            ),
            km.Buffer(
                CONSTANTS["kCbCol"], "bfloat16", CONSTANTS["kTileBytes"], window * pages
            ),
            km.Buffer(CONSTANTS["kCbOut"], "float32", 4096, 1),
        ],
        args=core_args,
        compile_args={
            "reader": row_tensor.accessor_args(),
            "writer": column_tensor.accessor_args() + output.accessor_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        handlers=KERNELS / "broadcast.h",
    )


def run_case(session, generator, stack, rect, args):
    """One load: `jobs` CAST commands on the flat group, each waited; every core of
    the rectangle against numpy."""
    group, core_args, cells = plan(session, stack, rect, args)
    _, _, width, height = rect
    blocks, pages = args.blocks, args.pages
    row_set, column_set = (
        generator.integers(-2, 3, (n, blocks, pages, TILE, TILE)).astype(np.float32)
        for n in (height, width)
    )

    def tensor(array, dtype):
        return session.tensor(array.reshape(-1, TILE), dtype, "tile", copy=True)

    row_tensor = tensor(row_set, "bfloat16")
    column_tensor = tensor(column_set, "bfloat16")
    output = tensor(
        np.zeros((args.jobs * group.count, TILE, TILE), np.float32), "float32"
    )
    programs = [
        program(one, row_tensor, column_tensor, output, args) for one in core_args
    ]
    if bands(session) is not None:  # a split card: one launch
        (programs,) = programs
    run = session.load(programs, deadline=args.deadline, wait=args.wait)
    log(f"{stack} {rect}: loaded")
    times, not_done, error = [], 0, None
    try:
        for job in range(args.jobs):
            words = [
                row_tensor.address,
                column_tensor.address,
                output.address,
                blocks,
                CONSTANTS["kNone"],
                0,
                job * group.count,
                blocks,
            ]
            start = time.perf_counter()
            ticket = run.submit(
                "broadcast", (group.first, group.count), words, exclusive=True
            )
            events = run.wait(ticket, timeout=5.0)
            times.append((time.perf_counter() - start) * 1e6)
            not_done += any(event.kind != "done" for event in events)
    except km.KomorebiError as caught:
        error = str(caught)
        log(f"{stack}: {error}")
    try:
        kernel_errors = [result["error_name"] for result in run.stop()]
    except km.KomorebiError as caught:
        kernel_errors = [str(caught)]
    want = row_set.sum((1, 2))[:, None] + column_set.sum((1, 2))[None, :]
    bad = []
    graph = session.graph
    for island, part in enumerate(output.parts):  # each island wrote its own copy
        got = part.read().reshape(args.jobs, group.count, TILE, TILE)
        first, count = graph.first(island), graph.count(island)
        bad += [
            {"job": job, "rank": rank}
            for rank, (row, column) in cells.items()
            if first <= rank < first + count
            for job in range(len(times))
            if not np.array_equal(got[job, rank], want[row, column])
        ]
    result = {
        "rect": list(rect),
        "jobs_done": len(times) - not_done,
        "us_a_round": round(float(np.median(times)), 2) if times else None,
        "cores_checked": len(cells) * len(times),
        "cores_bad": len(bad),
        "first_bad": bad[:8],
        "error": error,
        "kernel_errors": kernel_errors,
    }
    result["ok"] = (
        len(times) == args.jobs
        and not not_done
        and not bad
        and error is None
        and set(kernel_errors) == {"none"}
    )
    log(f"{stack}: {result}")
    return result


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    generator = np.random.default_rng(1)
    report = {"cards": cards, "blocks": args.blocks, "pages": args.pages}
    report["window"] = args.window
    links = 1 if len(cards) > 1 else 0
    opened = {"cards": cards, "links": links}
    if args.islands:  # one card split into islands of rows
        opened = {"card": cards[0], "islands": args.islands, "line": args.line}
        # its kernels with the page links' services, forwarding among them
        opened["kernel_bytes"] = 128 * 1024
        # its kernels read a tensor by its bank address: one copy in the card's DRAM
        opened["card_dram"] = True
    with km.Session(**opened, log=log) as session:
        grid = session.island(0).nocs().grid
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        report["cases"] = {
            stack: run_case(session, generator, stack, rect, args)
            for stack, rect in cases(session, grid)
        }
    report["ok"] = all(case["ok"] for case in report["cases"].values())
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
