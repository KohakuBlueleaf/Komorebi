"""X280 ports between X280 cores and Tensix cores, step by step (README.md): a tell
into a core, a core's tell to its X280 core, asks answered, the windows, a 4 KiB block,
a soak with DRAM reads, the two-way rate. Prints one JSON line per step."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench.power import aiclk_mhz

HERE = Path(__file__).resolve().parent
T0 = time.monotonic()
XP = {
    name: int(value, 0)
    for name, value in re.findall(
        r"#define (XP_\w+) (0x[0-9A-Fa-f]+|\d+)u", (HERE / "xp.h").read_text()
    )
}
STEPS = ("tell", "send", "ask", "window", "block", "soak", "timeout", "rate")


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def tensor(session, rows, cols):
    """A zeroed uint32 DRAM tensor of `rows` pages of `cols` words."""
    return session.tensor(np.zeros((rows, cols), np.uint32), "uint32")


def read(t):
    return t.read()


def program(words, clusters, serve, stats, reads, noc):
    """The grid's kernels, the thread on `clusters` and the ports."""
    prog = km.Program(
        "x280_ports",
        kernels=HERE / "kernels",
        buffers=[
            km.Buffer(
                index=16, format="uint32", page_bytes=XP["XP_SCRATCH_BYTES"], pages=1
            )
        ],
        compile_args={
            "reader": stats.accessor_args(),
            "writer": reads.accessor_args(),
        },
    )
    prog.thread(
        clusters=clusters,
        sources=[HERE / "x280" / "ports.c"],
        includes=[HERE],
        regions={"blocks": 132 * XP["XP_BLOCK_BYTES"]},  # a block per core, X280 core
        words=words,
        deadline=30.0,
    )
    prog.ports(serve=serve, noc=noc)
    return prog


def latency(pages, mhz):
    """Messages, wrong ones and the distribution in microseconds over the pages."""
    used = pages[pages[:, XP["XP_ST_COUNT"]] > 0]
    if len(used) == 0:
        return {"count": 0}
    first = XP["XP_ST_HIST"]
    hist = used[:, first : first + XP["XP_HIST_BINS"]].sum(axis=0, dtype=np.int64)
    cum = np.cumsum(hist) / hist.sum()
    edges = [b + 1 if b < 16 else (9 + (b - 16) % 8) << (1 + (b - 16) // 8)
             for b in range(XP["XP_HIST_BINS"])]  # fmt: skip
    top = int(used[:, XP["XP_ST_MAX"]].max())

    def pct(q):  # a bin's upper edge, at most the largest time seen
        return round(min(edges[int(np.searchsorted(cum, q))], top) / mhz, 3)

    total = (used[:, XP["XP_ST_SUM_HI"]].astype(np.int64) << 32) + used[
        :, XP["XP_ST_SUM_LO"]
    ]
    count = int(used[:, XP["XP_ST_COUNT"]].sum(dtype=np.int64))
    return {
        "count": count,
        "bad": int(used[:, XP["XP_ST_BAD"]].sum()),
        "gave": int((pages[:, XP["XP_ST_GAVE"]] != 0).sum()),
        "mean_microseconds": round(int(total.sum()) / count / mhz, 3),
        "min_microseconds": round(int(used[:, XP["XP_ST_MIN"]].min()) / mhz, 3),
        "p50_microseconds": pct(0.5),
        "p90_microseconds": pct(0.9),
        "p99_microseconds": pct(0.99),
        "p999_microseconds": pct(0.999),
        "max_microseconds": round(top / mhz, 3),
        "over_hist": int(hist[-1]),
        "core_mean_microseconds_max": round(
            float((total / used[:, XP["XP_ST_COUNT"]]).max()) / mhz, 3
        ),
    }


def cpus(results):
    """Each X280 core's counts in every cluster's result (answered, told, tells, done,
    ticks, passes, block transfer cycles)."""
    out = {}
    for c, r in results.items():
        w = list(r.words) + [0] * 44
        for h in range(4):
            row = w[16 + 7 * h : 23 + 7 * h]
            if any(row[:4]):
                out[f"{c}.{h}"] = row
        out[f"{c}.ports"] = w[44:47]  # tables' cores, cores served here, failures
    return out


def run_step(session, args, step, words, clusters, serve, noc=0):
    """Loads the program, waits for the threads, stops, reads the stats pages."""
    stats = tensor(session, 128, XP["XP_PAGE_BYTES"] // 4)
    reads = tensor(session, args.read_pages, XP["XP_READ_BYTES"] // 4)
    words = list(words)
    if step in ("ask", "soak", "timeout"):
        words[5], words[6], words[7] = (
            stats.address,
            reads.address,
            args.read_pages,
        )
    if step == "block":
        for i, b in ((5, 6), (6, 7)):  # the banks of clusters 1 and 2, 3
            bx, by = session.device.dram_core(b)
            words[i] = int(bx) | int(by) << 16
        words[7] = stats.address
    if step == "rate":
        words[4] = stats.address
    prog = program(words, clusters, serve, stats, reads, noc)
    run = session.load(prog, deadline=args.deadline, wait=args.wait)
    args.mhz = aiclk_mhz()  # with the grid running
    log(f"{step}: loaded, clusters {clusters}, words {words[:9]}")
    t = time.monotonic()
    results = run.thread_results(timeout=40.0)
    took = time.monotonic() - t
    for c, r in results.items():
        log(f"{step}: cluster {c}: {r.state}, code {r.code}, words {list(r.words)[:8]}")
    try:
        run.stop()
    except km.KomorebiError as err:  # a command that timed out ends the kernel
        log(f"{step}: the stop raised {type(err).__name__}: {err}")
    log(f"{step}: stopped after {took:.2f} s")
    pages = read(stats)
    return results, pages, took


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=STEPS)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--core", type=int, default=0)
    p.add_argument("--asks", type=int, default=100)
    p.add_argument("--clusters", type=int, nargs="+", default=[1])
    p.add_argument(
        "--one-x280", action="store_true", help="every core on X280 core 0 of cluster 1"
    )
    p.add_argument("--reads", action="store_true", help="the writers read DRAM")
    p.add_argument("--read-pages", type=int, default=4096)
    p.add_argument("--kind", choices=("l1", "dram"), default="l1")
    p.add_argument("--units", type=int, default=10000, help="rate: 1024-cycle units")
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--noc", type=int, default=0)
    p.add_argument("--deadline", type=float, default=1.0)
    p.add_argument("--wait", type=float, default=2.0)
    p.add_argument("--then", action="store_true", help="timeout: then a clean load")
    args = p.parse_args()
    out = {"step": args.step, "noc": args.noc}
    w = [0] * XP["XP_WORDS"]
    serve = None
    if args.step in ("tell", "send", "window"):
        mode = {"tell": "XP_MODE_TELL", "send": "XP_MODE_SEND"}.get(
            args.step, "XP_MODE_WINDOW"
        )
        w[:3] = [XP[mode], args.rounds, args.core]
        serve = {args.core: (1, 0)}
        args.clusters = [1]
    elif args.step == "block":
        kind = XP["XP_BLOCK_L1"] if args.kind == "l1" else XP["XP_BLOCK_DRAM"]
        w[:5] = [XP["XP_MODE_BLOCK"], args.rounds, args.cores, args.asks, kind]
    elif args.step == "rate":
        w[:4] = [XP["XP_MODE_RATE"], args.rounds, args.cores, args.units]
    else:
        flags = XP["XP_F_READS"] if args.reads else 0
        w[:5] = [XP["XP_MODE_ASK"], args.rounds, args.cores, args.asks, flags]
        w[8] = args.stop_after
    if args.step not in ("tell", "send", "window"):
        if args.one_x280:
            serve = {c: (args.clusters[0], 0) for c in range(args.cores)}
        else:
            hs = [(c, h) for c in args.clusters for h in range(4)]
            serve = {i: hs[i * len(hs) // args.cores] for i in range(args.cores)}
    with km.Session(log=log) as session:
        out["x280_mhz"] = session.pll_mhz
        log("device open")
        results, pages, took = run_step(
            session, args, args.step, w, args.clusters, serve, args.noc
        )
        mhz = out["aiclk_mhz"] = args.mhz
        lead = results[min(results)]
        out["code"] = lead.code
        out["state"] = lead.state
        out["words"] = list(lead.words)[:16]
        out["x280_cores"] = cpus(results)
        out["wall_s"] = round(took, 2)
        if args.step in ("ask", "soak", "timeout", "block", "rate"):
            out["latency"] = latency(pages[: args.cores], mhz)
            out["reads"] = int(pages[: args.cores, XP["XP_ST_READS"]].sum())
            out["core_sent"] = int(pages[: args.cores, XP["XP_ST_SENT"]].sum())
            out["core_seen"] = int(pages[: args.cores, XP["XP_ST_SEEN"]].sum())
        if args.step == "timeout" and args.then:
            log("timeout: the next load, the same session")
            w[8] = 0
            again, pages, _ = run_step(
                session, args, "ask", w, args.clusters, serve, args.noc
            )
            out["then_code"] = again[min(again)].code
            out["then_words"] = list(again[min(again)].words)[:8]
            out["then_latency"] = latency(pages[: args.cores], mhz)
    ok = out["code"] == 0
    if args.step in ("tell", "send"):
        ok = ok and out["words"][:2] == [args.rounds, 0]
    if args.step in ("ask", "soak", "block", "rate"):
        lat = out["latency"]
        ok = ok and out["words"][0] == args.rounds and lat.get("bad", 1) == 0
        ok = ok and lat.get("gave", 1) == 0
    if args.step == "timeout":
        ok = ok and out["words"][1] == 1
        if args.then:
            lat = out["then_latency"]
            ok = ok and out["then_code"] == 0 and out["then_words"][0] == args.rounds
            ok = ok and lat.get("bad", 1) == 0 and lat.get("gave", 1) == 0
    out["ok"] = bool(ok)
    print(json.dumps(out), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
