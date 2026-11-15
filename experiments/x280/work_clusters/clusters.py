"""Probes of X280 work clusters 1 to 3 beside the dispatcher: one probe image
(x280/probe.c) on several clusters at once, in modes sum, rates, push, ping and
parallel. One JSON line per run; README.md gives the commands."""

import argparse
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import komorebi as km  # noqa: E402
from komorebi import native  # noqa: E402
from komorebi import session as ks  # noqa: E402
from komorebi import thread as kt  # noqa: E402

KERNELS = ROOT / "tests" / "hardware" / "thread" / "kernels"
GO_LIM = 0x480
M_SUM, M_RATE, M_PUSH, M_PING, M_PARALLEL = 1, 2, 3, 4, 5
KINDS = {"shared": 0, "dram": 1, "lim": 2}
PING_LIM = 0x4C0  # a flag line in each cluster's LIM after the go line
T0 = time.monotonic()
TICK_MICROSECONDS = 1 / 50.0


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def pattern(seed32, first_word, n_words):
    """The probe's push and fill pattern: word i is seed ^ i * K (64-bit)."""
    seed = seed32 | seed32 << 32
    i = np.arange(first_word, first_word + n_words, dtype=np.uint64)
    return (i * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(seed)


def thread(cluster, words, regions, deadline=20.0):
    return kt.Thread(
        sources=[HERE / "x280" / "probe.c"],
        regions=regions,
        words=words,
        cluster=cluster,
        deadline=deadline,
    )


def regions(session, t):
    """A thread's own regions in a tensor of their own (the session's placement)."""
    return ks._Regions(session, t.bank, [t], {}).view(t)


def dram_xy(device, bank):
    x, y = device.dram_core(bank)
    return int(x) | int(y) << 16


class Clusters:
    """A session with a persistent grid and the dispatcher; probe threads started
    on their clusters through the port, released together by a go line."""

    def __init__(self, session):
        self.session = session
        self.port = None
        self.go = random.getrandbits(31) | 1

    def start(self, threads):
        """`threads`: {cluster: Thread}; regions written before the grid starts."""
        s = self.session
        self.threads = threads
        self.regions = {c: regions(s, t) for c, t in threads.items()}
        log(
            "regions written: " + str({c: hex(r.base) for c, r in self.regions.items()})
        )
        prog = km.Program("clusters", kernels=KERNELS)
        self.run = s.load(prog)
        self.port = self.run.backend.port
        log("grid and dispatcher ready")

    def go_all(self, timeout):
        """Starts every image, writes the go lines, waits; returns per cluster the
        result block (parsed) and the core results."""
        image = str(native.thread_image(next(iter(self.threads.values()))))
        for c, t in self.threads.items():
            r = self.regions[c]
            block = kt.args_block(t, r.base, r.check, table=r.table())
            block += bytes(64 * 3)  # the go line and the ping line start zeroed
            self.port.work_start(c, image, block)
        log(f"started on clusters {list(self.threads)}")
        self.info = info(self.port, self.threads)
        line = struct.pack("<I", self.go) + bytes(60)
        for c in self.threads:
            self.port.work_write(c, GO_LIM, line)
        out = {}
        for c in self.threads:
            w = self.port.work_wait(c, timeout)
            raw = self.port.work_read(c, kt.layout()["KMX_LIM_RESULT"], 256)
            done = self.port.work_stop(c, 2.0)
            res = kt.parse_result(raw, ran=w["ready"])
            out[c] = {
                "ran": w["ready"],
                "state": res.state,
                "error": res.error,
                "code": res.code,
                "words": res.words,
                "microseconds": res.microseconds,
                "where": res.where,
                "cleaned": done["cleaned"],
                "failed": w["failed"] or done["failed"],
            }
            log(f"cluster {c}: {res.state} {res.error} code {res.code}")
        return out

    def stop(self):
        self.run.stop()
        for r in self.regions.values():
            r.readable = True
        log("run stopped")


def info(port, clusters):
    out = {}
    for c in clusters:
        i = port.work_info(c)
        out[c] = {k: i[k] for k in ("hw_index", "noc_x", "noc_y")}
        out[c]["bank"] = kt.BANK_OF_CLUSTER[c]
    return out


def cmd_sum(args, session, out):
    rng = np.random.default_rng(1)
    data = {
        c: rng.integers(0, 1 << 32, args.bytes // 4, dtype=np.uint32)
        for c in args.clusters
    }
    k = Clusters(session)
    k.start({c: thread(c, [M_SUM, k.go], {"data": data[c]}) for c in args.clusters})
    res = k.go_all(10.0)
    k.stop()
    out["clusters"] = k.info
    for c, r in res.items():
        got = r["words"][0] | r["words"][1] << 32 if len(r["words"]) >= 2 else None
        want = int(data[c].view(np.uint64).sum(dtype=np.uint64))
        r["sum_exact"] = got == want and r["words"][3] == c  # and kmx_cluster()
        r["sum_microseconds"] = (
            round(r["words"][2] * TICK_MICROSECONDS, 1) if len(r["words"]) > 2 else None
        )
    out["result"] = res
    return all(r["sum_exact"] and r["code"] == 0 and r["cleaned"] for r in res.values())


def cmd_rates(args, session, out):
    k = Clusters(session)
    w = [M_RATE, k.go, args.reps]
    k.start({c: thread(c, w, {"data": args.bytes}) for c in args.clusters})
    res = k.go_all(30.0)
    k.stop()
    out["clusters"] = k.info
    names = ["invalidate", "read", "write", "flush"]
    ok = True
    for r in res.values():
        ok = ok and r["code"] == 0 and r["cleaned"]
        reps = []
        for i in range(args.reps):
            t = r["words"][4 * i : 4 * i + 4]
            reps.append(
                {
                    n: {
                        "microseconds": round(x * TICK_MICROSECONDS, 1),
                        "GBps": round(args.bytes / (x * 20e-9) / 1e9, 3),
                    }
                    for n, x in zip(names, t)
                    if x
                }
            )
        r["reps"] = reps
        del r["words"]
    out["result"] = res
    return ok


def cmd_push(args, session, out):
    seed = random.getrandbits(31) | 1
    k = Clusters(session)
    dev = session.device
    target = 1 if args.bank == 6 else 2  # a region in that bank: a thread's, never run
    noc = dram_xy(dev, args.bank)
    w = [M_PUSH, k.go]
    th = thread(args.cluster, w, {"data": 4096})
    tgt = thread(target, [0], {"target": args.bytes})
    registers = {args.cluster: th}
    k.threads = registers
    k.regions = {args.cluster: regions(session, th)}
    treg = regions(session, tgt)
    dest = treg.address("target")[1]
    th.words[:] = [M_PUSH, k.go, noc, dest, args.bytes, seed, args.pull]
    prog = km.Program("clusters", kernels=KERNELS)
    k.run = session.load(prog)
    k.port = k.run.backend.port
    log(f"push target bank {args.bank} at {dest:#x}, tile {noc & 0xFFFF},{noc >> 16}")
    res = k.go_all(30.0)
    k.stop()
    treg.readable = True
    got = treg.read("target").view(np.uint64)
    want = pattern(seed, 0, args.bytes // 8)
    r = res[args.cluster]
    out["clusters"] = k.info
    out["target"] = {"bank": args.bank, "tile": [noc & 0xFFFF, noc >> 16], "addr": dest}
    words = r.pop("words")
    r["push_microseconds"] = round(words[0] * TICK_MICROSECONDS, 1)
    r["push_GBps"] = round(args.bytes / (words[0] * 20e-9) / 1e9, 3)
    r["pull_microseconds"] = round(words[1] * TICK_MICROSECONDS, 1)
    r["pull_GBps"] = (
        round(args.pull / (words[1] * 20e-9) / 1e9, 3) if args.pull else None
    )
    pull_sum = words[2] | words[3] << 32
    r["pull_exact"] = pull_sum == int(want[: args.pull // 8].sum(dtype=np.uint64))
    r["push_wrong_words"] = int((got != want).sum())
    out["result"] = res
    return (
        r["code"] == 0
        and r["cleaned"]
        and r["push_wrong_words"] == 0
        and r["pull_exact"]
    )


def cmd_ping(args, session, out):
    a, b = args.pair
    kind = KINDS[args.kind]
    seed = random.getrandbits(31) | 1
    k = Clusters(session)
    dev = session.device
    ta = thread(a, [0], {"data": 4096, "flag": 4096})
    tb = thread(b, [0], {"data": max(args.xbytes, 4096), "flag": 4096})
    ra, rb = regions(session, ta), regions(session, tb)
    k.threads, k.regions = {a: ta, b: tb}, {a: ra, b: rb}
    fa, fb = ra.address("flag")[1], rb.address("flag")[1]
    xaddr = rb.address("data")[1]
    if kind == KINDS["lim"]:
        mine_a = mine_b = PING_LIM
        peer_a = peer_b = PING_LIM
    else:
        mine_a, mine_b, peer_a, peer_b = fa, fb, fb, fa
    prog = km.Program("clusters", kernels=KERNELS)
    k.run = session.load(prog)
    k.port = k.run.backend.port
    if kind == KINDS["lim"]:
        ia, ib = k.port.work_info(a), k.port.work_info(b)
        noc_a = ib["virtual_x"] | ib["virtual_y"] << 16
        noc_b = ia["virtual_x"] | ia["virtual_y"] << 16
        xbytes = 0
    else:
        noc_a = dram_xy(dev, rb.bank)
        noc_b = dram_xy(dev, ra.bank)
        xbytes = args.xbytes
    common = [args.rounds, kind]
    ta.words[:] = [M_PING, k.go, 0, *common, mine_a, peer_a, noc_a, xaddr, xbytes, seed]
    tb.words[:] = [M_PING, k.go, 1, *common, mine_b, peer_b, noc_b, xaddr, xbytes, seed]
    res = k.go_all(30.0)
    k.stop()
    out["clusters"] = k.info
    out["kind"] = args.kind
    ok = all(r["code"] == 0 and r["cleaned"] for r in res.values())
    wa = res[a]["words"]
    wb = res[b]["words"]
    out["round_trip_microseconds"] = round(
        wa[0] * TICK_MICROSECONDS / max(args.rounds, 1), 3
    )
    if xbytes:
        want = int(pattern(seed, 0, xbytes // 8).sum(dtype=np.uint64))
        out["exchange_exact"] = (wb[2] | wb[3] << 32) == want
        ok = ok and out["exchange_exact"]
    out["result"] = res
    return ok


def cmd_parallel(args, session, out):
    k = Clusters(session)
    k.start(
        {
            args.cluster: thread(
                args.cluster, [M_PARALLEL, k.go, args.calls], {"data": 4096}
            )
        }
    )
    res = k.go_all(30.0)
    k.stop()
    r = res[args.cluster]
    w = r.pop("words")
    r["parallel_microseconds"] = round(w[0] * TICK_MICROSECONDS / max(args.calls, 1), 3)
    out["result"] = res
    return r["code"] == 0 and r["cleaned"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sum")
    s.add_argument("--clusters", type=int, nargs="+", required=True)
    s.add_argument("--bytes", type=int, default=65536)
    s = sub.add_parser("rates")
    s.add_argument("--clusters", type=int, nargs="+", required=True)
    s.add_argument("--bytes", type=int, default=1 << 20)
    s.add_argument("--reps", type=int, default=3)
    s = sub.add_parser("push")
    s.add_argument("--cluster", type=int, required=True)
    s.add_argument("--bank", type=int, required=True)
    s.add_argument("--bytes", type=int, default=1 << 20)
    s.add_argument("--pull", type=int, default=1 << 18)
    s = sub.add_parser("ping")
    s.add_argument("--pair", type=int, nargs=2, required=True)
    s.add_argument("--kind", choices=list(KINDS), default="shared")
    s.add_argument("--rounds", type=int, default=1000)
    s.add_argument("--xbytes", type=int, default=65536)
    s = sub.add_parser("parallel")
    s.add_argument("--cluster", type=int, default=1)
    s.add_argument("--calls", type=int, default=10000)
    args = p.parse_args()
    out = {"probe": args.cmd, "args": vars(args)}
    with km.Session(log=log) as session:
        out["pll_mhz"] = session.pll_mhz
        ok = globals()[f"cmd_{args.cmd}"](args, session, out)
    out["ok"] = bool(ok)
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out, default=str), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
