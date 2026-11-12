"""X280 thread probes on work cluster 1. One JSON line per run; see README.md.

python probe.py two --cycles 100 --ticks 500000 --jobs 8
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import komorebi as km  # noqa: E402
from komorebi import native  # noqa: E402

BUILD = HERE / "build"
KERNELS = HERE / "kernels"
MAPCOLLECT = ROOT / "examples" / "01_mapcollect" / "kernels"
CLUSTER = 1
T0 = time.monotonic()

# LIM layout of the probe images (x280/probe.h).
PR_STAMP, PR_GO, PR_RESULT = 0x100, 0x140, 0x200
AICLK = Path("/sys/class/tenstorrent/tenstorrent!0/tt_aiclk")
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI word in each cluster's DRAM bank
BANKS = 8
# The DRAM bank each cluster's Memory Port reaches, by NoC0 coordinate (measured by
# an earlier experiment; the dram probe checks it).
BANK_OF = {(8, 3): 5, (8, 9): 6, (8, 5): 7, (8, 7): 7}
DP_WRITE, DP_COPY, DP_DONE = 1, 2, 0xD0D0D0D0
M32 = 0xFFFFFFFF


def log(msg):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def block(*words):
    """An argument block: 64-bit words (word 0 is the run value), padded with zeros up
    to the result words, so the stamp and go lines of an earlier run are cleared."""
    raw = struct.pack(f"<{len(words) + 1}Q", 0, *words)
    if len(raw) > PR_STAMP:
        raise ValueError("the argument block runs into the stamp line")
    return raw + bytes(PR_RESULT - len(raw))


def read_u64(port, offset, n):
    return list(struct.unpack(f"<{n}Q", port.work_read(CLUSTER, offset, 8 * n)))


def stamps(port):
    """Each hart's last progress stamp."""
    return list(struct.unpack("<4I", port.work_read(CLUSTER, PR_STAMP, 16)))


def wait_stamp(port, where, timeout=10.0):
    """Polls hart 0's progress stamp until it reaches `where`; raises past timeout."""
    by = time.monotonic() + timeout
    while True:
        got = stamps(port)[0]
        if got >= where:
            return got
        if time.monotonic() > by:
            raise TimeoutError(f"hart 0 at stamp {got}, waited for {where}")
        time.sleep(0.005)


def go(port, step):
    port.work_write(CLUSTER, PR_GO, struct.pack("<I", step) + bytes(60))


def dp_word(seed, page, w):
    """kernels/dram_probe.h dp_word on numpy arrays (uint32 arithmetic)."""
    page = np.asarray(page, dtype=np.uint64)
    w = np.asarray(w, dtype=np.uint64)
    x = (seed * 0x9E3779B1 & M32) ^ ((page * 4096 + w + 1) * 0x85EBCA77 & M32)
    x = np.asarray(x, dtype=np.uint64)
    x ^= x >> 15
    return (x * 0x2C1B3C6D & M32).astype(np.uint32)


def bank_rows(pages, page_words, seed):
    """[BANKS * pages, page_words]: row r is page r // BANKS of bank r % BANKS, filled
    with dp_word(seed + bank, page, w)."""
    r = np.arange(BANKS * pages)[:, None]
    w = np.arange(page_words)[None, :]
    return dp_word_banked(seed, r, w)


def dp_word_banked(seed, r, w):
    out = np.empty(np.broadcast_shapes(r.shape, w.shape), dtype=np.uint32)
    for k in range(BANKS):
        rows = np.arange(k, out.shape[0], BANKS)
        out[rows] = dp_word((seed + k) & M32, rows[:, None] // BANKS, w)
    return out


def uint32_tensor(device, rows):
    """A row-major int32 DRAM tensor holding the bits of uint32 rows (interleaved:
    row r in bank r % BANKS)."""
    data = np.ascontiguousarray(np.asarray(rows, dtype=np.uint32).view(np.int32))
    return ttnn.from_torch(
        torch.from_numpy(data),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def read_rows(tensor):
    return ttnn.to_torch(tensor).numpy().view(np.uint32)


def tensix(device, io, args, page):
    """Runs kernels/dram_rw.cpp on core (0, 0) with `args`, and waits for it."""
    core = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
    )
    runtime = ttnn.RuntimeArgs()
    runtime[0][0] = [int(a) for a in args]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNELS / "dram_rw.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core,
        compile_time_args=[],
        defines=[],
        runtime_args=runtime,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.RISCV_0_default,
        ),
        compiler_include_paths=[str(KERNELS)],
    )

    def cb(index, size):
        return ttnn.CBDescriptor(
            total_size=size,
            core_ranges=core,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=index, data_format=ttnn.uint32, page_size=size
                )
            ],
        )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[kernel], semaphores=[], cbs=[cb(0, page), cb(1, 64)]
    )
    ttnn.generic_op(io, descriptor)
    ttnn.synchronize_device(device)


def report_done(tensor, row, timeout=5.0):
    """Polls the Tensix kernel's report line (row `row` of `tensor`) until it says
    DP_DONE: the data writes before it have landed. Raises past the timeout."""
    by = time.monotonic() + timeout
    while True:
        rep = read_rows(tensor)[row]
        if int(rep[4]) == DP_DONE:
            if rep[3]:
                raise RuntimeError("the Tensix kernel's NoC wait gave up")
            return
        if time.monotonic() > by:
            raise TimeoutError("the Tensix kernel's report did not arrive")
        time.sleep(0.01)


def mapcollect_program(device):
    """The map/collect example's program, for a resident grid to run NOP records."""
    x = ttnn.from_torch(
        torch.zeros((32, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return km.Program(
        "mapcollect",
        kernels=MAPCOLLECT,
        buffers=[
            km.Buffer(index=0, format="bfloat16", page_bytes=2048, pages=16),
            km.Buffer(index=16, format="bfloat16", page_bytes=2048, pages=2),
        ],
        compile_args={
            "reader": ttnn.TensorAccessorArgs(x).get_compile_time_args(),
            "writer": ttnn.TensorAccessorArgs(x).get_compile_time_args(),
        },
        compute={"math_fidelity": "HiFi4", "fp32_dest_acc_en": True},
        segments=MAPCOLLECT / "mapcollect.h",
    )


def two(args):
    """The dispatch kernel on cluster 0 under a resident grid, and the beat image
    started, run and stopped on cluster 1 `cycles` times with NOP jobs between."""
    rng = np.random.default_rng(args.seed)
    image = str(BUILD / "beat.bin")
    out = {"probe": "two", "cycles": args.cycles, "ticks": args.ticks}
    bad, start_s, wait_s, stop_s, passes = [], [], [], [], []
    with km.Session(log=log) as session:
        log("device open")
        program = mapcollect_program(session.device)
        run = session.load(program, deadline=1.0)
        port = run.backend.port
        cores = run.cores
        out["cluster"] = port.work_info(CLUSTER)
        log(f"kernel ready; cluster {CLUSTER}: {out['cluster']}")
        for c in range(args.cycles):
            loud = c < 3 or c % 10 == 0
            if loud:
                log(f"cycle {c}: start")
            t = time.perf_counter()
            port.work_start(CLUSTER, image, block(args.ticks))
            start_s.append(time.perf_counter() - t)
            for _ in range(args.jobs):
                count = int(rng.integers(1, cores + 1))
                run.issue(0, (int(rng.integers(0, cores - count + 1)), count))
            run.drain(timeout=10.0)
            t = time.perf_counter()
            res = port.work_wait(CLUSTER, args.ticks / 50e6 + 2.0)
            wait_s.append(time.perf_counter() - t)
            t = time.perf_counter()
            done = port.work_stop(CLUSTER, 2.0)
            stop_s.append(time.perf_counter() - t)
            counts = read_u64(port, PR_RESULT, 8)
            ok = (
                res["ready"]
                and done["cleaned"]
                and res["results"] == counts[:4]
                and (args.ticks == 0 or min(counts[:4]) > 0)
            )
            passes.append(counts[:4])
            if not ok:
                bad.append({"cycle": c, "wait": res, "stop": done, "counts": counts})
                log(f"cycle {c}: FAILED {bad[-1]}")
                break
            if loud:
                log(f"cycle {c}: stopped, passes {counts[:4]}")
        log("stopping the kernel")
        result = run.stop()
        log("kernel stopped, state read back")
        states = run.state()
    out["cycles_ok"] = len(passes) - len(bad)
    out["bad"] = bad[:3]
    out["kernel"] = {k: result[k] for k in ("error_name", "records", "events")}
    out["cores_magic_ok"] = sum(c["magic_ok"] for c in states)
    out["gave_up_cores"] = sum(1 for c in states if c["reader"] or c["writer"])
    out["passes_min"] = int(np.min(passes)) if passes else 0
    for name, s in (("start", start_s), ("wait", wait_s), ("stop", stop_s)):
        out[f"{name}_ms"] = round(1e3 * float(np.median(s)), 2) if s else None
    out["ok"] = (
        not bad
        and out["cycles_ok"] == args.cycles
        and result["error_name"] == "none"
        and out["cores_magic_ok"] == cores
        and out["gave_up_cores"] == 0
    )
    return out


def dram(args):
    """Region A: the host fills it, the X280 reads it, a Tensix core writes it, the
    X280 reads it without and with Flush64. Region B: the X280 writes it, a Tensix
    core copies it out without a hand-over, after Flush64, and after a second
    pattern and Flush64."""
    page, na, nb = args.page, args.pages, args.pages
    words = page // 4
    seeds = {"host": 0x1000, "tensix": 0x2000, "b1": 0x3000, "b2": 0x4000}
    out = {"probe": "dram", "page": page, "pages": na}
    with km.Session(log=log) as session:
        device = session.device
        log("device open")
        ta = uint32_tensor(device, bank_rows(na + 1, words, seeds["host"]))
        tb = uint32_tensor(device, bank_rows(nb, words, seeds["host"]))
        outs = [
            uint32_tensor(device, np.zeros((BANKS * (nb + 1), words))) for _ in range(3)
        ]
        for t, n in [(ta, na + 1), (tb, nb)] + [(o, nb + 1) for o in outs]:
            at = t.buffer_address()
            if at <= IPI_ACK_DDR < at + n * page:
                raise RuntimeError(f"a buffer at {at:#x} covers the X280 IPI word")
        log("buffers written; tensix warm-up (compile) launch")
        warm = [DP_COPY, 0, tb.buffer_address(), 1, page, outs[2].buffer_address(), 0]
        tensix(device, [ta, tb, outs[2]], warm, page)
        log("tensix warm-up done")
        port = native.load_module().Dispatch(device, 1750)
        try:
            info = port.work_info(CLUSTER)
            bank = BANK_OF[(info["noc_x"], info["noc_y"])]
            out["cluster"] = info | {"expected_bank": bank}
            out["a_addr"], out["b_addr"] = ta.buffer_address(), tb.buffer_address()
            blk = block(
                ta.buffer_address(),
                tb.buffer_address(),
                na,
                nb,
                page,
                seeds["host"],
                seeds["tensix"],
                seeds["b1"],
                seeds["b2"],
                bank,
                int(20 * 50e6),
            )
            log("image start")
            port.work_start(CLUSTER, str(BUILD / "dram.bin"), blk)
            steps = [
                (
                    2,
                    [DP_WRITE, bank, ta.buffer_address(), na, page, 0, seeds["tensix"]],
                    ta,
                ),
                (
                    4,
                    [
                        DP_COPY,
                        bank,
                        tb.buffer_address(),
                        nb,
                        page,
                        outs[0].buffer_address(),
                        0,
                    ],
                    outs[0],
                ),
                (
                    5,
                    [
                        DP_COPY,
                        bank,
                        tb.buffer_address(),
                        nb,
                        page,
                        outs[1].buffer_address(),
                        0,
                    ],
                    outs[1],
                ),
                (
                    6,
                    [
                        DP_COPY,
                        bank,
                        tb.buffer_address(),
                        nb,
                        page,
                        outs[2].buffer_address(),
                        0,
                    ],
                    outs[2],
                ),
            ]
            for i, (where, targs, dst) in enumerate(steps):
                log(f"waiting for stamp {where}")
                wait_stamp(port, where)
                log(f"tensix step {i + 1} launch")
                tensix(device, [ta, tb, dst], targs, page)
                at = (na if i == 0 else nb) * BANKS + bank
                report_done(dst, at)
                log(f"tensix step {i + 1} done (report read back)")
                go(port, i + 1)
            log("waiting for the image")
            res = port.work_wait(CLUSTER, 5.0)
            got = struct.unpack("<16I", port.work_read(CLUSTER, PR_RESULT, 64))
            done = port.work_stop(CLUSTER, 2.0)
            log(f"image stopped: {res['ready']} {done['cleaned']}")
        finally:
            port.close()
            log("x280 runtime closed")
        rows_a = read_rows(ta)
        rows_o = [read_rows(o) for o in outs]
        log("buffers read back")
    names = ["status", "seen_bank", "a1_host", "a2_tensix", "a2_host", "a3_tensix",
             "a3_host", "flush_ticks", "flush_lines", "b_store_ticks", "words",
             "fault", "step"]  # fmt: skip
    out["x280"] = dict(zip(names, got))
    out["image"] = {"ready": res["ready"], "cleaned": done["cleaned"],
                    "results": res["results"], "failed": res["failed"]}  # fmt: skip
    w = np.arange(words)[None, :]

    def seen(rows, seed, n):
        mine = rows[bank::BANKS][:n]
        return int((mine == dp_word(seed, np.arange(n)[:, None], w)).sum())

    host_b = (seeds["host"] + bank) & M32
    out["tensix_reads_b"] = [
        {"x280_b1": seen(r, seeds["b1"], nb), "x280_b2": seen(r, seeds["b2"], nb),
         "host": seen(r, host_b, nb)}  # fmt: skip
        for r in rows_o
    ]
    out["tensix_wrote_a"] = seen(rows_a, seeds["tensix"], na)
    rep = rows_a[bank::BANKS][na][:5]
    noc = int(rep[0]) | int(rep[1]) << 32
    out["tensix_report"] = {
        "noc_addr": hex(noc),
        "noc_x": noc >> 36 & 63,
        "noc_y": noc >> 42 & 63,
        "local": hex(noc & ((1 << 36) - 1)),
        "bank_offset": int(rep[2]),
        "failed": int(rep[3]),
        "done": int(rep[4]) == DP_DONE,
    }
    return out


def allocate(device, rows, words):
    """An uninitialised row-major int32 DRAM tensor, interleaved over the banks."""
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, words]),
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )


def aiclk():
    """Card 0's Tensix clock in MHz, as the driver reports it (None if unreadable)."""
    try:
        return int(AICLK.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def check_ipi(tensors_and_bytes):
    """Refuses buffers whose bank range covers the X280 runtime's IPI word."""
    for t, n in tensors_and_bytes:
        at = t.buffer_address()
        if at <= IPI_ACK_DDR < at + n:
            raise RuntimeError(f"a buffer at {at:#x} covers the X280 IPI word")


def prefix_reference(counts):
    """out[c][b]: every core's counts in bins before b, plus bin b on cores before c."""
    c = counts.astype(np.int64)
    totals = c.sum(0)
    before = np.cumsum(totals) - totals
    return (before[None, :] + np.cumsum(c, 0) - c).astype(np.uint32)


def rates(args):
    """Read, write and Flush64 rates over two region sizes, four harts, then the
    prefix-sum job over cores x bins counters, checked against numpy."""
    rng = np.random.default_rng(args.seed)
    sizes = [args.small, args.large]
    page = 1 << 16
    pages = max(1, max(sizes) // page)
    cores, bins = args.cores, args.bins
    out = {"probe": "rates", "sizes": sizes, "cores": cores, "bins": bins}
    with km.Session(log=log) as session:
        device = session.device
        log("device open")
        tr = allocate(device, BANKS * pages, page // 4)
        counts = rng.integers(0, 64, (BANKS * cores, bins)).astype(np.uint32)
        tc = uint32_tensor(device, counts)
        to = allocate(device, BANKS * cores, bins)
        check_ipi([(tr, pages * page), (tc, cores * bins * 4), (to, cores * bins * 4)])
        log("buffers ready")
        port = native.load_module().Dispatch(device, 1750)
        try:
            info = port.work_info(CLUSTER)
            bank = BANK_OF[(info["noc_x"], info["noc_y"])]
            out["cluster"] = info | {"bank": bank}
            blk = block(
                tr.buffer_address(),
                sizes[0],
                sizes[1],
                tc.buffer_address(),
                to.buffer_address(),
                cores,
                bins,
                args.reps,
                int(20 * 50e6),
            )
            out["aiclk_mhz"] = [aiclk()]
            log("image start")
            port.work_start(CLUSTER, str(BUILD / "rates.bin"), blk)
            res = port.work_wait(CLUSTER, 25.0)
            log(f"image returned: {res['ready']}")
            out["aiclk_mhz"].append(aiclk())
            got = struct.unpack("<64I", port.work_read(CLUSTER, PR_RESULT, 256))
            done = port.work_stop(CLUSTER, 2.0)
            log(f"image stopped: {done['cleaned']}")
        finally:
            port.close()
            log("x280 runtime closed")
        result = read_rows(to)[bank::BANKS][:cores]
        log("output read back")
    out["image"] = {"ready": res["ready"], "cleaned": done["cleaned"],
                    "results": res["results"], "failed": res["failed"]}  # fmt: skip
    out["status"] = got[0]
    names = ["flush_clean", "read_cold", "read_again", "write", "flush_dirty",
             "read_scalar"]  # fmt: skip
    for i, size in enumerate(sizes):
        if size == 0:
            continue
        ticks = got[8 + 6 * i : 14 + 6 * i]
        out[f"pass_{size >> 20}MB"] = {
            n: {"us": round(t / 50, 1), "GBps": round(size / (t / 50e6) / 1e9, 2)}
            for n, t in zip(names, ticks)
            if t
        }
    out["job_us"] = [round(t / 50, 1) for t in got[32 : 32 + args.reps]]
    out["job_take_us"] = [round(t / 50, 1) for t in got[40 : 40 + args.reps]]
    want = prefix_reference(counts[bank::BANKS][:cores])
    out["job_exact"] = bool(np.array_equal(result, want))
    out["job_wrong_words"] = int((result != want).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="probe", required=True)
    p = sub.add_parser("two", help="kernel and grid, an image on cluster 1")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--ticks", type=int, default=0, help="mtime ticks each hart counts")
    p.add_argument("--jobs", type=int, default=1, help="NOP jobs per cycle")
    p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("dram", help="DRAM visibility between Tensix and cluster 1")
    p.add_argument("--page", type=int, default=4096)
    p.add_argument("--pages", type=int, default=4, help="pages per region")
    p = sub.add_parser("rates", help="read and write rates, the prefix-sum job")
    p.add_argument("--small", type=int, default=1 << 20, help="bytes, first pass")
    p.add_argument("--large", type=int, default=0, help="bytes, second pass")
    p.add_argument("--cores", type=int, default=119)
    p.add_argument("--bins", type=int, default=1024)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    probes = {"two": two, "dram": dram, "rates": rates}
    out = probes[args.probe](args)
    out["wall_s"] = round(time.monotonic() - T0, 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
