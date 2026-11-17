"""Commands across islands from the card: a NOP for island 1's ranks given to island 0's
dispatcher alone; island 0's X280 thread sending MAP to every rank of both islands, a
FOLD to island 1's thread, and a HANG on island 1 that its deadline gives up. Example
01's kernels, unchanged. Prints one JSON line; see README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import islands
from komorebi.bench import power

HERE = Path(__file__).resolve().parent
MAPCOLLECT = HERE.parent / "01_mapcollect"
sys.path.insert(0, str(MAPCOLLECT))
from mapcollect import rank_sums  # noqa: E402 (example 01's sums, beside it)

KERNELS = MAPCOLLECT / "kernels"
TILE = 32
TICKS_PER_US = 50  # the X280 timer, 50 MHz
THREAD_HANDLERS = {"maps": 1, "ask": 2, "fold": 3, "hang": 4, "counts": 5}  # across.c
EVENT_TIMEOUT, EVENT_ISLAND = 1, 0x80  # KM_EVENT_TIMEOUT, KM_EVENT_ISLAND
BAD = 0xFFFFFFFF
NO_RANK = 0xFFFFFFFF  # MAP's slow_rank: none
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
        "--graph",
        type=islands.parse_pairs,
        default=None,
        help="island pairs a-b,c-d (a pair k times: k links); default: each to each",
    )
    parser.add_argument(
        "--far", type=int, default=1, help="nop: the island whose ranks it goes to"
    )
    parser.add_argument(
        "--links", type=int, default=0, help="the session's page links (0: all found)"
    )
    parser.add_argument(
        "--steps",
        default="nop,map,ask,hang,count",
        help="of nop, map, ask, hang, count, host-hang and hops (not by default)",
    )
    parser.add_argument("--nop-jobs", type=int, default=2000, help="per round")
    parser.add_argument("--rounds", type=int, default=3, help="NOP rounds")
    parser.add_argument("--maps", type=int, default=500, help="MAPs a thread sends")
    parser.add_argument("--asks", type=int, default=500, help="FOLDs a thread asks")
    parser.add_argument(
        "--counts", type=int, default=1000, help="COUNTs from the host, then a thread"
    )
    parser.add_argument(
        "--reset-link",
        action="store_true",
        help="count: reset island 0's link core 0, then the thread's COUNTs again",
    )
    return parser.parse_args()


def program(source, output, threads):
    """Example 01's kernels; with `threads`, across.c on work cluster 1."""
    made = km.Program(
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
    if threads:
        made.thread(
            cluster=1,
            sources=[HERE / "thread" / "across.c"],
            regions={"scratch": 64},
            handlers=THREAD_HANDLERS,
            defines={
                "kMap": f"{int(made.handler('map'))}u",
                "kHang": f"{int(made.handler('hang'))}u",
                "kCount": f"{int(made.handler('count'))}u",
            },
        )
    return made


def timed_nops(run, group, jobs):
    """Microseconds a NOP of one round of `jobs` to `group`, pipelined; every event
    done."""
    started = time.perf_counter()
    handles = run.submit_many(0, np.full(jobs, group[0]), np.full(jobs, group[1]))
    events = run.wait(list(handles), timeout=30.0)
    spent = (time.perf_counter() - started) * 1e6 / jobs
    if any(event.kind != "done" for event in events):
        raise RuntimeError(f"NOP to {group}: {sorted({e.kind for e in events})}")
    return spent


def ask_thread(run, handler, args, timeout=20.0):
    """A command to island 0's thread (cluster 1), waited for: its word."""
    event = run.wait(run.submit(handler, cluster=1, args=args, result=True), timeout)
    if event.kind != "done":
        raise RuntimeError(f"thread command {handler}: {event.kind}")
    return event.result


def counted(run, own):
    """Every flat rank's count of COUNTs, as its writer gives it (COUNT a0 1 with
    the result, one rank a command on its island's own run, in flat ranks)."""
    handles = []
    for island, (low, cores) in enumerate(own):
        part = run.island(island)
        handles.append(
            [
                part.submit("count", (rank, 1), [1], result=True)
                for rank in range(low, low + cores)
            ]
        )
    counts = []
    for island, mine in enumerate(handles):
        for event in run.island(island).wait(mine, timeout=30.0):
            if event.kind != "done":
                raise RuntimeError(f"COUNT read on island {island}: {event.kind}")
            counts.append(event.result)
    return np.asarray(counts, np.int64)


def count_check(run, own, ranks, group, rounds, reset_link=False):
    """COUNT `rounds` times on one flat `group` over both islands from the host (its
    split, every line a part), one EXCLUSIVE COUNT on it from the host, then `rounds`
    from island 0's thread (crossed): every rank of the group counts each once."""
    inside = np.zeros(ranks, bool)
    inside[group[0] : group[0] + group[1]] = True
    report = {"count_group": list(group), "counts": rounds}
    started = time.perf_counter()
    tickets = run.submit_many(
        "count", [group[0]], [group[1]], np.zeros((rounds, 1), np.uint32)
    )
    events = run.wait(list(tickets), timeout=30.0)
    report["host_count_us"] = round((time.perf_counter() - started) * 1e6 / rounds, 3)
    kinds = {event.kind for parts in events for event in parts}
    exclusive = run.wait(run.submit("count", group, [0], exclusive=True), 10.0)
    report["exclusive_done"] = all(event.kind == "done" for event in exclusive)
    after_host = counted(run, own)
    want = np.where(inside, rounds + 1, 0)
    report["host_counts_exact"] = bool(kinds == {"done"}) and bool(
        (after_host == want).all()
    )
    log(
        f"host: {rounds} COUNTs on {group}, {report['host_count_us']} us a command;"
        f" exclusive {report['exclusive_done']}; counts {sorted(set(after_host))}"
    )
    ticks = ask_thread(run.island(0), "counts", [*group, rounds], timeout=60.0)
    report["thread_count_us"] = per_command_us(ticks, rounds)
    after_thread = counted(run, own)
    want = np.where(inside, 2 * rounds + 1, 0)
    report["thread_counts_exact"] = bool((after_thread == want).all())
    log(
        f"thread: {rounds} COUNTs on {group}, {report['thread_count_us']} us a"
        f" command; counts {sorted(set(after_thread))}"
    )
    if reset_link:  # the lines link's core, between two rounds that cross on it
        core = run.graph.link_cores[0][0]
        reset = run.reset_link(0, 0)
        report["link_reset"] = {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in reset.items()
        }
        ticks = ask_thread(run.island(0), "counts", [*group, rounds], timeout=60.0)
        report["thread_count_after_reset_us"] = per_command_us(ticks, rounds)
        after_reset = counted(run, own)
        want = np.where(inside, 3 * rounds + 1, 0)
        report["thread_counts_exact"] = report["thread_counts_exact"] and bool(
            (after_reset == want).all()
        )
        log(
            f"after the reset of link core 0 (core {core}): {rounds} COUNTs,"
            f" {report['thread_count_after_reset_us']} us a command; counts"
            f" {sorted(set(after_reset))}"
        )
    return report


def gave_up(run):
    """Each island's dispatcher's count of the host's commands it gave up, as the host
    reads it."""
    return [run.island(i).dispatcher_given_up() for i in range(2)]


def host_hang(run, group, hung, before):
    """After the thread's HANG (no host command given up on either island), the same
    HANG from the host on island 0's run alone (no part mark: it crosses): given up
    naming island 1, island 0 counting it once and island 1 not, and the run goes on."""
    after_thread = gave_up(run)
    first = run.island(0)
    handle = first.submit("hang", group, [hung, 1])
    try:
        event = first.wait(handle, timeout=20.0)
        named, given = event.kind, False
    except km.CommandTimeout as error:
        named, given = error.slot & 0xFF, error.given_up
    after_host = gave_up(run)
    log(
        f"host HANG on {group} to island 0 alone: given up {given}, auxiliary"
        f" {named}; give-ups before {before}, after the thread's {after_thread},"
        f" after the host's {after_host}"
    )
    return {
        "host_hang_given_up": bool(given and named == EVENT_ISLAND + 1),
        "gave_up_counts": [before, after_thread, after_host],
        "gave_up_counts_agree": after_thread == before
        and after_host == [before[0] + 1, before[1]],
    }


def healthy_after(run, own, hung, words):
    """After the HANGs (their cores reset at the give-ups), commands on the other ranks:
    a NOP on each island's own ring, example 01's MAP from the host on each island's
    healthy ranks (its split), and a MAP from island 0's thread on island 1's healthy
    ranks (it crosses). Each is answered."""
    source, output = words
    low1, cores1 = own[1]
    after = hung + 10  # island 1's ranks past the hung one's HANG group
    groups = [own[0], (after, low1 + cores1 - after)]
    answered = {}
    for island, rank in ((0, own[0][0] + 10), (1, low1 + cores1 - 1)):
        part = run.island(island)
        event = part.wait(part.submit(0, (rank, 1)), timeout=10.0)
        answered[f"nop_island_{island}"] = event.kind == "done"
    for first, count in groups:
        args = [source, 2 * count, output, first, 0, NO_RANK, 2 * first]
        events = run.wait(run.submit("map", (first, count), args), timeout=10.0)
        answered[f"map_{first}_{count}"] = all(e.kind == "done" for e in events)
    first, count = groups[1]
    ticks = ask_thread(run.island(0), "maps", [first, count, 1, source, output])
    answered[f"thread_map_{first}_{count}"] = ticks != BAD
    log(f"after the HANGs, other ranks: {answered}")
    return {"healthy_after": answered, "healthy_answered": all(answered.values())}


def hops_check(run, graph, rounds, jobs):
    """From island 0's dispatcher alone (its own run: they cross): a NOP to each
    island's ranks; COUNTs on islands 2 on, each rank counting each; a HANG on the last
    island named; a NOP on islands 1 to the one before the last after it (the hung
    part's cores end at their wait limit, and a split card resets none)."""
    first = run.island(0)
    islands = len(graph.islands)
    own = [(graph.first(i), graph.count(i)) for i in range(islands)]
    report = {"hops_nop_us": {}}
    for island in range(1, islands):
        spent = float(
            np.median([timed_nops(first, own[island], jobs) for _ in range(3)])
        )
        report["hops_nop_us"][island] = round(spent, 3)
    log(f"NOP from island 0's dispatcher to each island: {report['hops_nop_us']} us")
    far = (own[2][0], graph.ranks - own[2][0])
    handles = [first.submit("count", far, [0]) for _ in range(rounds)]
    kinds = {event.kind for event in first.wait(handles, timeout=30.0)}
    counts = counted(run, own)
    want = np.zeros(graph.ranks, np.int64)
    want[far[0] :] = rounds
    report["hops_count_group"] = list(far)
    report["hops_counts_exact"] = kinds == {"done"} and bool((counts == want).all())
    log(f"{rounds} COUNTs on {far}: {kinds}, exact {report['hops_counts_exact']}")
    hung = graph.ranks - 3  # one of the last island's ranks
    try:
        event = first.wait(first.submit("hang", far, [hung - far[0], 1]), timeout=20.0)
        named = event.kind
    except km.CommandTimeout as error:
        named = error.slot & 0xFF
    report["hops_hang_named"] = named
    report["hops_hang_named_island"] = named == EVENT_ISLAND + islands - 1
    log(f"HANG on {far}, rank {hung} hung: named {named}")
    alive = (own[1][0], own[-1][0] - own[1][0])
    after = first.wait(first.submit(0, alive), timeout=10.0)
    report["hops_after_hang"] = after.kind == "done"
    log(f"a NOP on {alive} after it: {after.kind}")
    return report


def per_command_us(ticks, count):
    return None if ticks == BAD else round(ticks / TICKS_PER_US / count, 3)


def main():
    args = parse()
    steps = args.steps.split(",")
    cards = [int(card) for card in args.cards.split(",")]
    threads = any(
        step in steps for step in ("map", "ask", "hang", "count", "host-hang")
    )
    report = {"cards": cards, "steps": steps, "links": args.links}
    generator = np.random.default_rng(25)
    opened = (
        {"card": cards[0], "graph": args.graph} if args.islands else {"cards": cards}
    )
    with km.Session(
        **opened,
        islands=args.islands or None,
        links=args.links,
        log=log,
    ) as session:
        ranks = session.ranks
        graph = session.graph
        own = [(graph.first(i), graph.count(i)) for i in range(2)]
        log(f"open: {ranks} ranks, islands {own}")
        exact = generator.integers(-2, 3, (2 * ranks * TILE, TILE)).astype(np.float32)
        source = session.tensor(exact, "bfloat16", "tile")
        output = session.tensor(
            np.zeros((ranks * TILE, TILE), np.float32), "bfloat16", "tile"
        )
        made = [program(source, output, threads) for _ in range(2)]
        # a split card: one launch, so one program for every island
        run = session.load(made[0] if args.islands else made, deadline=2.0)
        log("loaded on both islands")
        first = run.island(0)
        if "nop" in steps:
            rounds = range(args.rounds)
            report["nop_local_us"] = float(
                np.median([timed_nops(first, own[0], args.nop_jobs) for _ in rounds])
            )
            far = (graph.first(args.far), graph.count(args.far))
            report["nop_far"] = args.far
            report["nop_across_us"] = float(
                np.median([timed_nops(first, far, args.nop_jobs) for _ in rounds])
            )
            log(
                f"NOP from island 0's ring: its own ranks {report['nop_local_us']:.3f}"
                f" us, island {args.far}'s {report['nop_across_us']:.3f} us"
            )
        wrong = None
        if "map" in steps:
            words = [source.address, output.address]
            local = ask_thread(first, "maps", [*own[0], args.maps, *words])
            whole = ask_thread(first, "maps", [0, ranks, args.maps, *words])
            report["thread_map_local_us"] = per_command_us(local, args.maps)
            report["thread_map_all_us"] = per_command_us(whole, args.maps)
            log(
                f"thread MAP: island 0's ranks {report['thread_map_local_us']} us, "
                f"all {ranks} {report['thread_map_all_us']} us"
            )
            wrong = "map"
        if "ask" in steps:
            number = 3 * 1 + 1  # KMX_THREAD(1, 1)
            word = ask_thread(first, "ask", [number, args.asks, 0x1234])
            report["thread_ask_us"] = per_command_us(word, args.asks)
            report["ask_word"] = word
            log(f"thread to thread: {report['thread_ask_us']} us a FOLD ({word:#x})")
        if "hang" in steps:
            group, hung = (100, 40), 30  # rank 130: one of island 1's
            word = ask_thread(first, "hang", [*group, hung, 300000, 2000000])
            report["hang_word"] = word
            report["hang_named_island"] = word == (
                EVENT_TIMEOUT << 8 | EVENT_ISLAND + 1
            )
            log(f"HANG on {group}: word {word:#x}")
            words = [source.address, output.address]
            after = ask_thread(first, "maps", [0, ranks, 1, *words])
            report["map_after_hang"] = after != BAD
            log(f"a MAP on all ranks after it: {report['map_after_hang']}")
            wrong = "map"
        if "count" in steps:
            report.update(
                count_check(
                    run, own, ranks, (60, 120), args.counts, reset_link=args.reset_link
                )
            )
        if "hops" in steps:
            report.update(hops_check(run, graph, args.counts, args.nop_jobs))
        if "host-hang" in steps:
            before = gave_up(run)
            word = ask_thread(first, "hang", [100, 40, 30, 300000, 2000000])
            log(f"thread HANG on (100, 40): word {word:#x}")
            report.update(host_hang(run, (100, 40), 30, before))
            words = (source.address, output.address)
            report.update(healthy_after(run, own, 130, words))
        log("stopping")
        results = run.stop()
        report["kernel_errors"] = [  # a split card resets no given-up core: named
            "ended, named" if "ended" in result else result["error_name"]
            for result in results
        ]
        log(f"stopped: {report['kernel_errors']}")
        if wrong:
            (got,) = session.read_bf16(output)
            report.update(check(exact, got, ranks))
            log(f"output: {report['wrong_tiles']} wrong of {ranks}")
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    report["ok"] = bool(
        all(error in ("none", "ended, named") for error in report["kernel_errors"])
        and report.get("wrong_tiles", 0) == 0
        and report.get("thread_map_all_us", 0) is not None
        and report.get("thread_ask_us", 0) is not None
        and report.get("hang_named_island", True)
        and report.get("map_after_hang", True)
        and report.get("host_hang_given_up", True)
        and report.get("gave_up_counts_agree", True)
        and report.get("healthy_answered", True)
        and report.get("exclusive_done", True)
        and report.get("host_counts_exact", True)
        and report.get("thread_counts_exact", True)
        and report.get("hops_counts_exact", True)
        and report.get("hops_hang_named_island", True)
        and report.get("hops_after_hang", True)
    )
    print(json.dumps(power.clocked(report)), flush=True)
    sys.exit(0 if report["ok"] else 1)


def check(exact, got, ranks):
    """Every rank's tile of the last MAP on all ranks against example 01's float64
    sums, bit-exact (integer inputs)."""
    sums = np.cumsum(exact.reshape(-1, TILE, TILE).astype(np.float64), 0)
    prefix = np.concatenate([np.zeros((1, TILE, TILE)), sums])
    want = rank_sums(prefix, 0, 2 * ranks, ranks)
    wrong = int((got.reshape(-1, TILE, TILE) != want).any(axis=(1, 2)).sum())
    return {"tiles_checked": ranks, "wrong_tiles": wrong}


if __name__ == "__main__":
    main()
