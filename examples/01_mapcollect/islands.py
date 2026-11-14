"""Map/collect over cards as the islands of one session (`Session(cards=...)`), or one
card split into islands (`--islands`): the same program and kernels on every island,
flat groups split by island, an input split over the islands; checks the tiles with
mapcollect.py's sums, times commands, prints a JSON line. See docs/reference/islands.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from mapcollect import rank_sums

import komorebi as km
from komorebi import formats
from komorebi.errors import describe_core

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
TILE = 32
NO_RANK = 0xFFFFFFFF
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0,1")
    parser.add_argument(
        "--islands",
        type=int,
        default=0,
        help="split the one card of --cards into this many islands (2 to 4)",
    )
    parser.add_argument("--nop-rounds", type=int, default=3)
    parser.add_argument("--nop-jobs", type=int, default=20000, help="per round")
    parser.add_argument("--map-rounds", type=int, default=3)
    parser.add_argument("--map-jobs", type=int, default=2000, help="per round")
    parser.add_argument(
        "--whole",
        action="store_true",
        help="MAP on one group of every rank, else on each island's own group",
    )
    parser.add_argument(
        "--hang", action="store_true", help="a HANG on island 1 must name its rank"
    )
    parser.add_argument(
        "--across",
        action="store_true",
        help="MAP on island 0 reading an input island 1 holds",
    )
    parser.add_argument(
        "--reset-link",
        type=int,
        default=-1,
        metavar="ISLAND",
        help="with --across: reset that island's link core 0, then read across again",
    )
    parser.add_argument(
        "--put",
        action="store_true",
        help="MAP on every rank storing into an output island 0 holds",
    )
    return parser.parse_args()


def map_words(source, output, groups):
    """MAP arguments of each group (first, count): rank r of the group adds input
    tiles 2r and 2r + 1 of the group's own window and stores page first + r."""
    groups = np.asarray(groups, np.int64)
    jobs = len(groups)
    return np.stack(
        [
            np.full(jobs, source.address),
            2 * groups[:, 1],  # tiles: two a rank
            np.full(jobs, output.address),
            groups[:, 0],  # first_page: the group's first rank
            np.zeros(jobs, np.int64),  # slow_cycles
            np.full(jobs, NO_RANK),  # slow_rank: none
            2 * groups[:, 0],  # first_tile
        ]
    ).T


def timed(run, handler, groups, words=None):
    """Microseconds a command of one round: submit_many of the groups, then drain."""
    groups = np.asarray(groups, np.int64)
    start = time.perf_counter()
    run.submit_many(handler, groups[:, 0], groups[:, 1], words)
    run.drain(timeout=30.0)
    return (time.perf_counter() - start) * 1e6 / len(groups)


def median(values):
    return round(float(np.median(values)), 3)


def open_session(args, cards):
    """Two or more cards as islands, or one card split into `--islands` islands."""
    if args.islands:
        (card,) = cards
        return km.Session(card=card, islands=args.islands, log=log)
    # the write call's put path takes the kernels past the runtime's 69 KiB buffer
    kernel_bytes = 96 * 1024 if args.put else None
    return km.Session(cards=cards, log=log, kernel_bytes=kernel_bytes)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "split": args.islands}
    generator = np.random.default_rng(1)
    with open_session(args, cards) as session:
        ranks = session.ranks
        report["ranks"] = ranks
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        log(f"open: {ranks} ranks, islands {report['islands']}")
        exact = generator.integers(-2, 3, (2 * ranks * TILE, TILE)).astype(np.float32)
        source = session.tensor(exact, "bfloat16", "tile")
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        report["source_parts"] = list(source.placement.parts)
        log(f"input split {report['source_parts']} at {source.address:#x}")
        (back,) = session.read_bf16(source)
        report["source_read_exact"] = bool(np.array_equal(back, exact))
        log(f"input read back from every island: {report['source_read_exact']}")
        far = source
        if args.across:
            far = session.tensor(exact, "bfloat16", "tile", like=report["islands"][1])
            log(f"input held by island 1 alone: {list(far.placement.parts)}")
        held = output
        if args.put:
            held = session.tensor(
                np.zeros((ranks * TILE, TILE), np.float32),
                "bfloat16",
                "tile",
                like=report["islands"][0],
            )
            log(f"output held by island 0 alone: {list(held.placement.parts)}")
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                # the placements of MAP's inputs (a7): the split one, the far one
                "reader": source.accessor_args()
                + source.placement_args()
                + far.placement_args(),
                # the placements of MAP's outputs (a8): the split one, the held one
                "writer": output.accessor_args()
                + output.placement_args()
                + held.placement_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program, deadline=1.0)
        log("loaded on every island")
        if args.across:
            report.update(across(session, run, args, source, far, output, exact))
        elif args.put:
            report.update(put(session, run, args, source, output, held, ranks))
            output = held  # the last round's
        else:
            report.update(commands(session, run, args, source, output, ranks))
        if args.hang:
            report.update(hang(session, run, source, ranks, exact))
        log("stopping")
        try:
            results = run.stop()
            report["kernel_errors"] = [result["error_name"] for result in results]
            report["kernel_commands"] = [result["commands"] for result in results]
        except km.KomorebiError as error:
            report["stop_error"] = str(error)
        log(f"stopped: {report.get('kernel_errors', report.get('stop_error'))}")
        (got,) = session.read_bf16(output)
        log("output read back from every island")
        report.update(check(exact, got, report["map_groups"]))
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["source_read_exact"]
        and report["wrong_tiles"] == 0
        and report.get("kernel_errors", ["stop"]) == ["none"] * len(report["islands"])
        and report.get("hang_named_island", True)
        and report.get("read_island1_in_run_exact", True)
        and report.get("crossings_agree", True)
        and report.get("before_reset", {}).get("wrong_tiles", 0) == 0
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def commands(session, run, args, source, output, ranks):
    """NOP rounds on the first island alone and on every rank; MAP rounds on each
    island's own group (or every rank with --whole); joined tickets."""
    out = {}
    graph = session.graph
    own = [(graph.first(i), graph.count(i)) for i in range(len(session.sessions))]
    rounds, jobs = args.nop_rounds, args.nop_jobs
    out["nop_island_us"] = [
        median([timed(run, 0, [group] * jobs) for _ in range(rounds)]) for group in own
    ]
    log(f"NOP on each island's own ranks {own}: {out['nop_island_us']} us")
    out["nop_all_us"] = median(
        [timed(run, 0, [(0, ranks)] * jobs) for _ in range(rounds)]
    )
    log(f"NOP on all {ranks} ranks: {out['nop_all_us']} us")
    tickets = [run.submit(0, (0, ranks)) for _ in range(4)]
    events = run.wait(tickets, timeout=10.0)
    out["joined_ticket_parts"] = [
        [(event.kind, event.auxiliary) for event in parts] for parts in events
    ]
    log(f"joined tickets: {out['joined_ticket_parts']}")
    groups = [(0, ranks)] if args.whole else own
    handler = run.program.handler("map")
    alone = [groups[0]] * args.map_jobs
    out["map_one_group_us"] = median(
        [
            timed(run, handler, alone, map_words(source, output, alone))
            for _ in range(args.map_rounds)
        ]
    )
    log(f"MAP on group {groups[0]}: {out['map_one_group_us']} us")
    every = [groups[job % len(groups)] for job in range(args.map_jobs)]
    out["map_groups_us"] = median(
        [
            timed(run, handler, every, map_words(source, output, every))
            for _ in range(args.map_rounds)
        ]
    )
    out["map_groups"] = [list(group) for group in groups]
    log(f"MAP on groups {groups} in turn: {out['map_groups_us']} us")
    return out


def hang(session, run, source, ranks, exact):
    """A HANG on a group over both islands, addressed and reported by flat rank: an
    inspect of the hanging rank, then the deadline's give-up; island 1's pages read."""
    out = {}
    graph = session.graph
    # Island 1's rank 10 in a group over both islands. Its reader waits for a port
    # arrival (hang_mode 1) until the deadline gives it up.
    rank = graph.first(1) + 10
    named = f"rank {rank} (island 1, core {graph.core_of(rank)[1]})"
    group = (graph.first(1) - 20, graph.count(1) + 20)
    hang = run.submit("hang", group, [rank - group[0], 1])
    run.send()
    time.sleep(0.25)  # a quarter of the deadline
    (core,) = run.inspect((rank, 1))
    out["inspect_rank"] = describe_core(core)
    log(f"inspect of rank {rank}: {out['inspect_rank']}")
    try:
        run.wait(hang, timeout=10.0)
        out["hang"] = "returned"
    except km.CommandTimeout as error:
        out["hang"] = str(error)[:400]
        out["hang_recovered"] = error.recovered
        out["hang_named_island"] = (
            error.island == 1 and error.given_up and named in str(error)
        )
    log(f"HANG on group {group}: {out['hang']}")
    out["hang_named_island"] = out.get("hang_named_island", False) and (
        out["inspect_rank"].startswith(named)
    )
    raw, _ = formats.to_bytes(exact, "bfloat16", "tile")
    rows = raw.reshape(source.pages, source.page_bytes)
    (got,) = session.read(source, first=ranks, pages=16)
    out["read_island1_in_run_exact"] = bool(
        np.array_equal(got, rows[ranks : ranks + 16])
    )
    log(f"16 pages of island 1 read in the run: {out['read_island1_in_run_exact']}")
    return out


def across(session, run, args, local, far, output, exact):
    """Rounds of one MAP on island 0's ranks (its own run: island 0 starts at rank 0)
    reading `local`, then `far` (island 1 holds it; the reader asks for it itself);
    what crossed checked against the rounds' asks and pages; the last round checked
    (--reset-link: checked, that island's link core 0 reset, read across again)."""
    graph = session.graph
    group = (graph.first(0), graph.count(0))
    handler = run.program.handler("map")
    count = group[1]

    def one_round(source, placed):
        words = map_words(source, output, [group])[0].tolist() + [placed]
        started = time.perf_counter()
        run.wait(run.submit(handler, group, words), timeout=10.0)
        return (time.perf_counter() - started) * 1e6

    out = {"map_groups": [list(group)], "links": session.links}
    tile_bytes = 2 * count * 2048
    for name, source, placed in (("local", local, 0), ("across", far, 1)):
        before = run.crossings()
        spent = median([one_round(source, placed) for _ in range(args.map_rounds)])
        moved = run.crossings() - before
        out[f"map_{name}_us"] = spent
        out[f"map_{name}_gb_s"] = round(tile_bytes / spent / 1e3, 3)
        log(f"MAP on {group} reading {name}: {spent} us a round, {tile_bytes:,} B")
        log(f"crossed while reading {name}:\n{moved}")
        out[f"crossed_{name}"] = {
            f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))
        }
    # each rank asks once a round for its two tiles; island 1 sends them
    asks, pages = args.map_rounds * count, args.map_rounds * 2 * count
    went, came = out["crossed_across"]["0->1"], out["crossed_across"]["1->0"]
    out["crossings_agree"] = (
        went["asks"] == asks
        and came["pages"] == pages
        and came["bytes"] == pages * far.aligned_page_bytes
        and not any(out["crossed_local"][way]["asks"] for way in ("0->1", "1->0"))
    )
    log(f"asks {went['asks']} of {asks}, pages {came['pages']} of {pages}")
    if args.reset_link >= 0:
        (got,) = session.read_bf16(output)
        out["before_reset"] = check(exact, got, [group])
        session.write(output, np.zeros((output.pages, output.page_bytes), np.uint8))
        reset = run.reset_link(args.reset_link, 0)
        out["link_reset"] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in reset.items()
        }
        out["map_across_after_reset_us"] = round(one_round(far, 1), 3)
        log(f"after the reset, MAP across: {out['map_across_after_reset_us']} us")
    return out


def put(session, run, args, source, output, held, ranks):
    """Rounds of one MAP on every rank (one flat group over the islands) storing into
    `output` (split: each rank's page on its own island), then into `held` (island 0
    holds every page): island 1's ranks put theirs across with the same writer's
    write call, which returns once the page is written there. The last round is
    checked."""
    group = (0, ranks)
    handler = run.program.handler("map")

    def one_round(target, placed):
        words = map_words(source, target, [group])[0].tolist() + [0, placed]
        started = time.perf_counter()
        run.wait(run.submit(handler, group, words), timeout=10.0)
        return (time.perf_counter() - started) * 1e6

    out = {"map_groups": [list(group)], "links": session.links}
    for name, target, placed in (("split", output, 0), ("put", held, 1)):
        spent = median([one_round(target, placed) for _ in range(args.map_rounds)])
        out[f"map_{name}_us"] = spent
        log(f"MAP on {group} storing {name}: {spent} us a round")
    return out


def check(exact, got, groups):
    """Every rank's tile of the last MAP round against mapcollect.py's float64 sums,
    bit-exact (integer inputs)."""
    sums = np.cumsum(exact.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    got = got.reshape(-1, TILE, TILE)
    wrong, checked, ranks = 0, 0, []
    for first, count in groups:
        want = rank_sums(prefix, 2 * first, 2 * count, count)
        bad = (got[first : first + count] != want).any(axis=(1, 2))
        wrong += int(bad.sum())
        ranks += (first + np.flatnonzero(bad)).tolist()
        checked += count
    out = {"tiles_checked": checked, "wrong_tiles": wrong, "wrong_ranks": ranks[:8]}
    if ranks:  # the first wrong tile: how far off, and whether it is zeros
        first = ranks[0]
        want = rank_sums(prefix, 2 * first, 2, 1)[0]
        out["first_wrong"] = {
            "max_error": float(np.abs(got[first] - want).max()),
            "zeros": bool(not got[first].any()),
            "rows_wrong": int((got[first] != want).any(axis=1).sum()),
        }
    return out


if __name__ == "__main__":
    main()
