"""Map/collect over cards as the islands of one session (`Session(cards=...)`), or one
card split into islands (`--islands`): the same program and kernels on every island,
flat groups, the input placed by a hint (`--split`, `--group`); checks the tiles with
mapcollect.py's sums, times commands and writes. See docs/reference/islands.md."""

import argparse
import concurrent.futures
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
        "--split", type=int, default=None, help="the input's hint: split dimension"
    )
    parser.add_argument(
        "--group",
        default=None,
        metavar="FIRST,RANKS",
        help="the input's hint: the group that splits it (default every rank)",
    )
    parser.add_argument(
        "--io-mib",
        default="64",
        metavar="MIB,...",
        help="sizes written and read over every island at once, then island 0 alone",
    )
    parser.add_argument(
        "--across",
        action="store_true",
        help="MAP on island 0 reading an input island 1 holds",
    )
    parser.add_argument(
        "--per-rank", type=int, default=2, help="--across: input tiles a rank, 1 to 4"
    )
    parser.add_argument(
        "--check-rounds",
        action="store_true",
        help="--across: check every round's tiles, not only the last's",
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


def map_words(source, output, groups, per=2):
    """MAP arguments of each group (first, count): rank r of the group adds input
    tiles per r .. per r + per - 1 of the group's own window, stores page first + r.
    The tensors by their reference words."""
    groups = np.asarray(groups, np.int64)
    jobs = len(groups)
    return np.stack(
        [
            np.full(jobs, source.address),
            per * groups[:, 1],  # tiles: `per` a rank
            np.full(jobs, output.address),
            groups[:, 0],  # first_page: the group's first rank
            np.zeros(jobs, np.int64),  # slow_cycles
            np.full(jobs, NO_RANK),  # slow_rank: none
            per * groups[:, 0],  # first_tile
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
    # the page calls over islands take the kernels past the runtime's 69 KiB buffer
    return km.Session(cards=cards, log=log, kernel_bytes=96 * 1024)


def main():
    args = parse()
    cards = [int(card) for card in args.cards.split(",")]
    report = {"cards": cards, "split": args.islands, "per_rank": 2}
    generator = np.random.default_rng(1)
    with open_session(args, cards) as session:
        ranks = session.ranks
        report["ranks"] = ranks
        report["islands"] = [
            [session.graph.first(i), session.graph.count(i)]
            for i in range(len(session.sessions))
        ]
        log(f"open: {ranks} ranks, islands {report['islands']}")
        # rank r's input tiles 2r and 2r + 1 are row r of a [ranks, 2] tile matrix
        exact = generator.integers(-2, 3, (2 * ranks, TILE, TILE)).astype(np.float32)
        matrix = exact.reshape(ranks, 2, TILE, TILE).transpose(0, 2, 1, 3)
        matrix = matrix.reshape(ranks * TILE, 2 * TILE)
        hint = {}
        if args.split is not None:
            hint["split"] = args.split
        if args.group is not None:
            hint["group"] = tuple(int(value) for value in args.group.split(","))
        report["hint"] = hint
        source = session.tensor(matrix, "bfloat16", "tile", **hint)
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        report["source_parts"] = list(source.placement.parts)
        report["source_part_pages"] = [
            source.placement.part_pages(i) for i in range(len(session.sessions))
        ]
        log(
            f"input {hint or 'no hint'}: steps {report['source_parts']}, pages "
            f"{report['source_part_pages']}, word {source.address:#x}"
        )
        (back,) = session.read_bf16(source)
        report["source_read_exact"] = bool(np.array_equal(back, matrix))
        log(f"input read back from every island: {report['source_read_exact']}")
        far = source
        if args.across:
            far = session.tensor(
                matrix, "bfloat16", "tile", group=tuple(report["islands"][1])
            )
            log(f"input held by island 1 alone: {list(far.placement.parts)}")
        held = output
        if args.put:
            held = session.tensor(
                np.zeros((ranks * TILE, TILE), np.float32),
                "bfloat16",
                "tile",
                group=tuple(report["islands"][0]),
            )
            log(f"output held by island 0 alone: {list(held.placement.parts)}")
        moved = {}
        for mib in (int(size) for size in args.io_mib.split(",")):
            shape = (mib * (1 << 20) // 2048 * TILE, TILE)
            moved[mib] = {
                "every island": session.empty(shape, "bfloat16", "tile"),
                "island 0": session.empty(
                    shape, "bfloat16", "tile", group=tuple(report["islands"][0])
                ),
            }
        program = km.Program(
            "mapcollect",
            kernels=KERNELS,
            buffers=[
                km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
                km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
            ],
            compile_args={
                "reader": source.accessor_args(),
                "writer": output.accessor_args(),
            },
            compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
            handlers=KERNELS / "mapcollect.h",
        )
        run = session.load(program, deadline=1.0)
        log("loaded on every island")
        report["io"] = io_rates(session, moved, generator)
        report["channel"] = channel_rates(session, generator)
        if args.across:
            report["per_rank"] = args.per_rank
            report.update(across(session, run, args, source, far, output, exact))
        elif args.put:
            report.update(put(session, run, args, source, output, held, ranks))
            output = held  # the last round's
        else:
            report.update(commands(session, run, args, source, output, ranks))
        if args.hang:
            report.update(hang(session, run, source, ranks, matrix))
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
        report.update(check(exact, got, report["map_groups"], report["per_rank"]))
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        report["source_read_exact"]
        and report["wrong_tiles"] == 0
        and not report.get("rounds_wrong")
        and report.get("kernel_errors", ["stop"]) == ["none"] * len(report["islands"])
        and report.get("hang_named_island", True)
        and report.get("read_island1_in_run_exact", True)
        and report.get("crossings_agree", True)
        and all(io["exact"] for io in report.get("io", []))
        and report.get("before_reset", {}).get("wrong_tiles", 0) == 0
    )
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)


def io_rates(session, sizes, generator):
    """GB/s of one write and one read of each tensor (random bytes, the run loaded):
    over every island's host channel at once, then over island 0's alone; the read
    compared whole; each island's last round on its cores (cycles)."""
    out = []
    for mib, tensors in sizes.items():
        for name, tensor in tensors.items():
            pages = generator.integers(
                0, 256, (tensor.pages, tensor.page_bytes), np.uint8
            )
            row = {"mib": mib, "held by": name}
            started = time.perf_counter()
            session.write(tensor, pages)
            row["write_s"] = round(time.perf_counter() - started, 4)
            row["write_cycles"] = [one.transfer_cycles for one in session.sessions]
            started = time.perf_counter()
            (got,) = session.read(tensor)
            row["read_s"] = round(time.perf_counter() - started, 4)
            row["read_cycles"] = [one.transfer_cycles for one in session.sessions]
            row["write_gb_s"] = round(pages.nbytes / row["write_s"] / 1e9, 3)
            row["read_gb_s"] = round(pages.nbytes / row["read_s"] / 1e9, 3)
            row["exact"] = bool(np.array_equal(got, pages))
            log(f"{mib} MiB held by {name}: {row}")
            out.append(row)
    return out


def channel_rates(session, generator, mib=64):
    """GB/s of the host's own copies into and out of the islands' host channels (no
    transfer): island 0's alone, then each island's from its own thread at once."""
    data = generator.integers(0, 256, mib << 20, np.uint8)
    channels = [one.channel[: mib << 20] for one in session.sessions]
    halves = np.split(data, len(channels))

    def copies(work):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(len(work)) as pool:
            for done in [pool.submit(*one) for one in work]:
                done.result()
        return round(data.nbytes / (time.perf_counter() - started) / 1e9, 3)

    def into(channel, part):
        channel[: part.nbytes] = part

    def out_of(channel, part):
        part[:] = channel[: part.nbytes]

    back = np.empty_like(data)
    out = {
        "into_island0_gb_s": copies([(into, channels[0], data)]),
        "out_of_island0_gb_s": copies([(out_of, channels[0], back)]),
        "into_each_gb_s": copies(
            [(into, channel, half) for channel, half in zip(channels, halves)]
        ),
        "out_of_each_gb_s": copies(
            [
                (out_of, channel, half)
                for channel, half in zip(channels, np.split(back, len(channels)))
            ]
        ),
    }
    log(f"host copies of {mib} MiB, host channels: {out}")
    return out


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


def hang(session, run, source, ranks, matrix):
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
    raw, _ = formats.to_bytes(matrix, "bfloat16", "tile")
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
    count, per = group[1], args.per_rank
    if not 1 <= per <= 4:
        raise ValueError(f"--per-rank {per}: 1 to 4 (the input holds 2 tiles a rank)")
    out = {"map_groups": [list(group)], "links": session.links, "rounds_wrong": []}

    def one_round(name, source, number):
        words = map_words(source, output, [group], per)[0].tolist()
        started = time.perf_counter()
        run.wait(run.submit(handler, group, words), timeout=10.0)
        spent = (time.perf_counter() - started) * 1e6
        if args.check_rounds:  # this round's tiles, each wrong one named
            (got,) = session.read_bf16(output)
            wrong = check(exact, got, [group], per)
            if wrong["wrong_tiles"]:
                out["rounds_wrong"].append({"phase": name, "round": number, **wrong})
                log(f"{name} round {number}: {wrong}")
        return spent

    tile_bytes = per * count * 2048
    for name, source in (("local", local), ("across", far)):
        before = run.crossings()
        spent = median([one_round(name, source, n) for n in range(args.map_rounds)])
        moved = run.crossings() - before
        out[f"map_{name}_us"] = spent
        out[f"map_{name}_gb_s"] = round(tile_bytes / spent / 1e3, 3)
        log(f"MAP on {group} reading {name}: {spent} us a round, {tile_bytes:,} B")
        log(f"crossed while reading {name}:\n{moved}")
        out[f"crossed_{name}"] = {
            f"{a}->{b}": moved.total(a, b) for a, b in ((0, 1), (1, 0))
        }
    # each rank asks once a round for its tiles; island 1 sends them
    asks, pages = args.map_rounds * count, args.map_rounds * per * count
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
        out["before_reset"] = check(exact, got, [group], per)
        session.write(output, np.zeros((output.pages, output.page_bytes), np.uint8))
        reset = run.reset_link(args.reset_link, 0)
        out["link_reset"] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in reset.items()
        }
        out["map_across_after_reset_us"] = round(one_round("after reset", far, 0), 3)
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

    def one_round(target):
        words = map_words(source, target, [group])[0].tolist()
        started = time.perf_counter()
        run.wait(run.submit(handler, group, words), timeout=10.0)
        return (time.perf_counter() - started) * 1e6

    out = {"map_groups": [list(group)], "links": session.links}
    for name, target in (("split", output), ("put", held)):
        spent = median([one_round(target) for _ in range(args.map_rounds)])
        out[f"map_{name}_us"] = spent
        log(f"MAP on {group} storing {name}: {spent} us a round")
    return out


def check(exact, got, groups, per=2):
    """Every rank's tile of the last MAP round against mapcollect.py's float64 sums,
    bit-exact (integer inputs); the first 8 wrong ones named (named_wrong)."""
    tiles = exact.reshape(-1, TILE, TILE).astype(np.float64)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), np.cumsum(tiles, 0)])
    got = got.reshape(-1, TILE, TILE).astype(np.float64)
    wrong, checked, named = 0, 0, []
    for first, count in groups:
        want = rank_sums(prefix, per * first, per * count, count)
        bad = (got[first : first + count] != want).any(axis=(1, 2))
        wrong += int(bad.sum())
        for index in np.flatnonzero(bad)[: 8 - len(named)].tolist():
            rank = first + index
            named.append(named_wrong(tiles, got[rank], want[index], rank, per))
        checked += count
    return {"tiles_checked": checked, "wrong_tiles": wrong, "wrong": named}


def named_wrong(tiles, got, want, rank, per):
    """Rank `rank`'s wrong sum of input tiles per rank ..: for each of its pages, what
    a wrong page alone would have been (another input tile, zeros); else the rows."""
    out = {"rank": rank, "rows_wrong": int((got != want).any(axis=1).sum())}
    for page in range(per):
        tile = per * rank + page
        instead = tiles[tile] + got - want  # that page as the core summed it
        if not instead.any():
            out[f"page {tile}"] = "zeros"
            continue
        same = np.flatnonzero((tiles == instead).all(axis=(1, 2))).tolist()
        if same:
            out[f"page {tile}"] = f"input tile {same[0]}" + (
                f" (and {len(same) - 1} more)" if len(same) > 1 else ""
            )
    return out


if __name__ == "__main__":
    main()
