"""An X280 core as a rank of a dynamic run: messages to its queue in its cluster's
region, replies into the cores' inboxes in L1, the dispatcher's own end; port asks
beside them. Prints one JSON line. See README.md."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import abi
from komorebi.bench.power import aiclk_mhz
from komorebi.program import thread

HERE = Path(__file__).resolve().parent
STARTED = time.monotonic()
XR = {
    name: int(value, 0)
    for name, value in re.findall(
        r"#define (XR_\w+) (0x[0-9A-Fa-f]+|\d+)u", (HERE / "xr.h").read_text()
    )
}
X280_TICKS_PER_SECOND = 50_000_000
ROLE_RANK, ROLE_STAND_IN, ROLE_NONE = 0, 1, 2


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def l1_area(session, grid, words):
    """A zeroed L1 area of `words` words on every core, at one address."""
    cores = grid[0] * grid[1]
    return session.tensor(
        np.zeros((cores, words), np.uint32), "uint32", shard_grid=grid
    )


def plan(args):
    """The X280 cores used, each rank core's X280 core, the stand-ins: ({core: (cluster,
    X280 core)}, {core: (role, stand-in)}, [(cluster, X280 core)])."""
    x280 = [(cluster, cpu) for cluster in (1, 2, 3) for cpu in range(4)][: args.x280]
    serve, roles = {}, {}
    for core in range(args.ranks):
        index = core % len(x280)
        serve[core] = x280[index]
        roles[core] = (ROLE_RANK, args.ranks + index)
    for index, owner in enumerate(x280):
        stand_in = args.ranks + index
        serve[stand_in] = owner
        roles[stand_in] = (ROLE_STAND_IN, stand_in)
    return serve, roles, x280


def thread_words(session, inbox, x280, ranks):
    """The thread's words (xr.h)."""
    headers = abi.headers()
    words = [0] * XR["XR_WORDS"]
    words[XR["XR_WORD_INBOX"]] = inbox
    words[XR["XR_WORD_IDLE"]] = 750  # 15 microseconds, the grid's 20,000 cycles
    for cluster in (1, 2, 3):
        words[XR["XR_WORD_BACK"] + cluster - 1] = (
            headers["KMD_LIM_BYTES"]
            + (cluster - 1) * thread.ring_bytes()
            - headers["KMD_LIM_MESSAGES"]
        )
        x, y = session.device.dram_core(thread.BANK_OF_CLUSTER[cluster])
        words[XR["XR_WORD_BANK"] + cluster - 1] = int(x) | int(y) << 16
    for cluster in (1, 2, 3):
        for cpu in range(4):
            words[XR["XR_WORD_STANDIN"] + 4 * (cluster - 1) + cpu] = XR["XR_NONE"]
    for index, (cluster, cpu) in enumerate(x280):
        words[XR["XR_WORD_STANDIN"] + 4 * (cluster - 1) + cpu] = ranks + index
    return words


def program(session, args, areas):
    """The grid's kernels as a dynamic program, the thread and the ports."""
    serve, roles, x280 = plan(args)
    inbox, work, stats = areas

    def core_args(core):
        role, stand_in = roles.get(core, (ROLE_NONE, 0))
        return [role, stand_in, inbox.address, work.address, stats.address], [], []

    made = km.Program(
        "x280_rank",
        kernels=HERE / "kernels",
        args=core_args,
        handlers={"run": XR["XR_HANDLER_RUN"], "finish": XR["XR_HANDLER_FINISH"]},
    ).dynamic(
        "run",
        message_bytes=64,
        max_live=1024,
        queue_messages=1024,
        bins=1,
        idle_cycles=20000,
        counters=1,
    )
    made.thread(
        clusters=sorted({cluster for cluster, _ in x280}),
        sources=[HERE / "x280" / "rank.c"],
        includes=[HERE],
        regions={"queues": 4 * XR["XR_AREA"]},
        words=thread_words(session, inbox.address, x280, args.ranks),
        deadline=55.0,
    )
    made.ports(serve=serve)
    return made, x280


def latency(pages, mhz):
    """Round trips, wrong ones and their distribution in microseconds over the pages."""
    used = pages[pages[:, XR["XR_ST_COUNT"]] > 0]
    if len(used) == 0:
        return {"count": 0}
    first = XR["XR_ST_HIST"]
    hist = used[:, first : first + XR["XR_HIST_BINS"]].sum(axis=0, dtype=np.int64)
    cumulative = np.cumsum(hist) / hist.sum()
    edges = [b + 1 if b < 16 else (9 + (b - 16) % 8) << (1 + (b - 16) // 8)
             for b in range(XR["XR_HIST_BINS"])]  # fmt: skip
    top = int(used[:, XR["XR_ST_MAX"]].max())

    def at(quantile):  # a bin's upper edge, at most the largest time seen
        return round(
            min(edges[int(np.searchsorted(cumulative, quantile))], top) / mhz, 3
        )

    total = (used[:, XR["XR_ST_SUM_HI"]].astype(np.int64) << 32) + used[
        :, XR["XR_ST_SUM_LO"]
    ]
    count = int(used[:, XR["XR_ST_COUNT"]].sum(dtype=np.int64))
    return {
        "count": count,
        "mean_microseconds": round(int(total.sum()) / count / mhz, 3),
        "min_microseconds": round(int(used[:, XR["XR_ST_MIN"]].min()) / mhz, 3),
        "p50_microseconds": at(0.5),
        "p99_microseconds": at(0.99),
        "max_microseconds": round(top / mhz, 3),
    }


def wall(pages, low):
    return (pages[:, low + 1].astype(np.int64) << 32) + pages[:, low]


def summary(pages, mhz, trips, host_seconds):
    """One RUN's pages (rank cores): checks, round trips, rates."""
    done = int(pages[:, XR["XR_ST_DONE"]].sum(dtype=np.int64))
    first, last = wall(pages, XR["XR_ST_FIRST_LO"]), wall(pages, XR["XR_ST_LAST_LO"])
    span = int(last.max() - first.min()) / mhz  # microseconds, the grid's wall clock
    sends = int(pages[:, XR["XR_ST_SENDS"]].sum(dtype=np.int64))
    send_cycles = int(pages[:, XR["XR_ST_SEND_CYCLES"]].sum(dtype=np.int64))
    sent = int(pages[:, XR["XR_ST_SENT"]].sum(dtype=np.int64))
    return {
        "round_trips": done,
        "expected": trips * len(pages),
        "bad": int(pages[:, XR["XR_ST_BAD"]].sum()),
        "extra": int(pages[:, XR["XR_ST_EXTRA"]].sum()),
        "gave": [int(g) for g in pages[:, XR["XR_ST_GAVE"]] if g],
        "sent": sent,
        "consumed": int(pages[:, XR["XR_ST_CONSUMED"]].sum(dtype=np.int64)),
        "idle_lines": int(pages[:, XR["XR_ST_IDLES"]].sum(dtype=np.int64)),
        "credits": int(pages[:, XR["XR_ST_CREDITS"]].sum(dtype=np.int64)),
        "latency": latency(pages, mhz),
        "span_microseconds": round(span, 1),
        "round_trips_per_second_grid_clock": (
            round(done / span * 1e6) if span > 0 else 0
        ),
        "round_trips_per_second_host": (
            round(done / host_seconds) if host_seconds else 0
        ),
        "host_seconds": round(host_seconds, 4),
        "send_microseconds_per_message": (
            round(send_cycles / mhz / sent, 3) if sent else None
        ),
        "messages_per_send": round(sent / sends, 2) if sends else None,
    }


def x280_cores(results, x280, x280_mhz):
    """Each X280 core's result words, by "cluster.core"."""
    out = {}
    for cluster, cpu in x280:
        words = list(results[cluster].words) + [0] * XR["XR_RES_WORDS"]
        at = XR["XR_RES_CORE"] + XR["XR_RES_STRIDE"] * cpu
        row = words[at : at + XR["XR_RES_STRIDE"]]
        consumed = max(row[XR["XR_RES_CONSUMED"]], 1)
        out[f"{cluster}.{cpu}"] = {
            "consumed": row[XR["XR_RES_CONSUMED"]],
            "sent": row[XR["XR_RES_SENT"]],
            "bad": row[XR["XR_RES_BAD"]],
            "gave": row[XR["XR_RES_GAVE"]],
            "asks": row[XR["XR_RES_ASKS"]],
            "take_microseconds": round(row[XR["XR_RES_TAKE"]] * 16 / x280_mhz, 1),
            "take_microseconds_per_message": round(
                row[XR["XR_RES_TAKE"]] * 16 / x280_mhz / consumed, 3
            ),
            "store_microseconds_per_message": round(
                row[XR["XR_RES_STORE"]] * 16 / x280_mhz / consumed, 3
            ),
            "messages_per_take": round(consumed / max(row[XR["XR_RES_TAKES"]], 1), 2),
            "empty_looks": row[XR["XR_RES_POLLS"]],
            "runs": row[XR["XR_RES_RUNS"]],
        }
    return out


def steps(args):
    """(name, window, mode) of each RUN of the load."""
    out = [(f"rank_w{window}", window, XR["XR_MODE_RANK"]) for window in args.windows]
    if args.ask:
        out.append(("ask", 1, XR["XR_MODE_ASK"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ranks", type=int, default=1, help="rank cores of the grid")
    parser.add_argument("--x280", type=int, default=1, help="X280 cores, 1 to 12")
    parser.add_argument("--trips", type=int, default=100000, help="round trips in all")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--ask", action="store_true", help="the port asks beside")
    parser.add_argument("--deadline", type=float, default=10.0)
    args = parser.parse_args()
    if not 1 <= args.x280 <= 12 or args.ranks < args.x280:
        raise SystemExit("1 to 12 X280 cores, at least one rank core each")
    if len(steps(args)) > XR["XR_PAGES"]:
        raise SystemExit(f"at most {XR['XR_PAGES']} runs")
    trips = -(-args.trips // args.ranks)  # per rank core
    out = {
        "ranks": args.ranks,
        "x280": args.x280,
        "trips_per_core": trips,
        "load": Path("/proc/loadavg").read_text().split()[:3],
    }
    ok = True
    with km.Session(log=log) as session:
        grid = session.grid
        if args.ranks + args.x280 > grid[0] * grid[1]:
            raise SystemExit("more cores than the grid has")
        areas = (
            l1_area(session, grid, XR["XR_INBOX"] * 16),
            l1_area(session, grid, (XR["XR_WORK_STAGE"] + 64 * XR["XR_INBOX"]) // 4),
            l1_area(session, grid, XR["XR_PAGES"] * XR["XR_PAGE_BYTES"] // 4),
        )
        made, x280 = program(session, args, areas)
        run = session.load(made, deadline=args.deadline, wait=2.0)
        out["x280_mhz"] = session.pll_mhz
        clocks = [aiclk_mhz()]
        log(f"loaded: {args.ranks} rank cores, X280 cores {x280}")
        group = (0, args.ranks + args.x280)
        events = {}
        for page, (name, window, mode) in enumerate(steps(args)):
            started = time.perf_counter()
            handle = run.submit(
                "run", group, [trips, window, mode, page], exclusive=True
            )
            event = run.wait(handle, timeout=args.deadline + 5.0)
            events[name] = (event.kind, time.perf_counter() - started)
            log(f"{name}: {event.kind} after {events[name][1]:.4f} s")
            ok = ok and event.kind == "done"
        clocks.append(aiclk_mhz())
        run.wait(run.submit("finish", (args.ranks, args.x280)), timeout=5.0)
        results = run.thread_results(timeout=20.0)
        for cluster, result in results.items():
            log(f"cluster {cluster}: {result.state}, code {result.code}")
            ok = ok and result.state == "done" and result.code == 0
        stopped = run.stop()
        log("stopped")
        out["dispatcher_error"] = stopped["error_name"]
        ok = ok and stopped["error"] == 0
        stats = areas[2].read().reshape(grid[0] * grid[1], XR["XR_PAGES"], -1)
        mhz = out["aiclk_mhz"] = int(round(np.median(clocks)))
        out["aiclk_samples"] = clocks
        runs = {}
        for page, (name, _, mode) in enumerate(steps(args)):
            pages = stats[: args.ranks, page, :].astype(np.uint32)
            runs[name] = summary(pages, mhz, trips, events[name][1])
            runs[name]["event"] = events[name][0]
            log(f"{name}: {runs[name]}")
            exact = runs[name]["round_trips"] == runs[name]["expected"]
            exact = exact and runs[name]["bad"] == 0 and runs[name]["extra"] == 0
            exact = exact and not runs[name]["gave"]
            if mode == XR["XR_MODE_RANK"]:
                exact = exact and runs[name]["sent"] == runs[name]["consumed"]
            ok = ok and exact
        out["runs"] = runs
        out["x280_cores"] = x280_cores(results, x280, session.pll_mhz)
        x280_sent = sum(core["sent"] for core in out["x280_cores"].values())
        x280_consumed = sum(core["consumed"] for core in out["x280_cores"].values())
        grid_sent = sum(r["sent"] for r in runs.values())
        grid_consumed = sum(r["consumed"] for r in runs.values())
        out["totals"] = {
            "grid_sent": grid_sent,
            "x280_consumed": x280_consumed,
            "x280_sent": x280_sent,
            "grid_consumed": grid_consumed,
        }
        ok = ok and grid_sent == x280_consumed and x280_sent == grid_consumed
        ok = ok and all(
            core["bad"] == 0 and core["gave"] == 0
            for core in out["x280_cores"].values()
        )
    out["ok"] = bool(ok)
    print(json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
