"""Overlays: kernel bodies that a persistent kernel reads from DRAM into a fixed L1
slot at run time and calls. One question per job; progress lines, then one JSON line.

Example: python probe.py capture; python probe.py swap --reps 1000"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "build" / "overlay_probe"
# One JIT cache for every job: the loader's ELF the overlays link against is the one
# that runs. Set before ttnn is imported.
os.environ["TT_METAL_CACHE"] = str(OUT / "jit-cache")
sys.path.insert(0, str(ROOT / "src"))

import builder  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from komorebi import native  # noqa: E402
from komorebi.session import KERNEL_BUFFER_BASE, Session, aiclk_mhz  # noqa: E402

T0 = time.monotonic()
CTL_WORDS = 64  # control words per core (overlay.h, loader.cpp)
RESP_WORDS = 16
OP_RUN, OP_ALT, OP_EXIT, OP_COMPUTE, OP_CLOCK = 1, 2, 3, 4, 5
FLAG_INVALIDATE, FLAG_AT = 1, 2
LEAD_S = 0.02  # a scheduled broadcast starts this long after the clock was read
STATUS = {
    0: "ok",
    1: "bad op",
    2: "read timeout",
    3: "too big",
    4: "idle",
    5: "compute",
}
SLOT_BYTES = 96 * 1024
TSLOT_BYTES = 16 * 1024  # each compute core's slot
IDLE_UNITS = 2000  # the loader leaves after this many 2^20 cycles without a command
COMMANDS = OUT / "build-commands.txt"
TILE_BYTES = 2048  # a bfloat16 tile
CB_IN, CB_OUT = 0, 16  # compute.cpp, scalar.cpp, loader.cpp
TRISCS = ("trisc0", "trisc1", "trisc2")  # unpack, math, pack
# File-static library state a compute part keeps its own copy of and resets at entry
RESETS = ("_ZN7ckernel4mathL16src_zero_flag_hwE",)


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def core_set(cores):
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]
    )


def l1_rows(device, cores, words):
    """An int32 L1 tensor with one shard of `words` words on each core."""
    config = ttnn.create_sharded_memory_config(
        (1, words),
        core_set(cores),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return dram_rows(device, np.zeros((len(cores), words), np.uint32), config)


def dram_rows(device, rows, config=None):
    """An int32 tensor holding the bits of the uint32 rows (DRAM: one page a row)."""
    data = np.ascontiguousarray(np.asarray(rows, np.uint32).view(np.int32))
    return ttnn.from_torch(
        torch.from_numpy(data),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=config or ttnn.DRAM_MEMORY_CONFIG,
    )


def page_place(tensor, page):
    """(bank, address) of an interleaved DRAM tensor's page."""
    aligned = tensor.buffer_aligned_page_size()
    return page % 8, tensor.buffer_address() + (page // 8) * aligned


def loader_descriptor(cores, ctl, slot, tslots, results):
    args = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        bank, at = page_place(results, i)
        args[x][y] = [
            ctl.buffer_address(),
            bank,
            at,
            slot.buffer_address(),
            SLOT_BYTES,
            IDLE_UNITS,
            tslots.buffer_address() if tslots is not None else 0,
            TSLOT_BYTES,
        ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "loader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set(cores),
        compile_time_args=[],
        runtime_args=args,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.RISCV_0_default,
        ),
        compiler_include_paths=[str(HERE)],
    )
    if tslots is None:
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
    targs = ttnn.RuntimeArgs()
    for x, y in cores:
        targs[x][y] = [
            ctl.buffer_address(),
            tslots.buffer_address(),
            TSLOT_BYTES,
            IDLE_UNITS,
        ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set(cores),
        compile_time_args=[],
        runtime_args=targs,
        config=ttnn.ComputeConfigDescriptor(),
        compiler_include_paths=[str(HERE)],
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * TILE_BYTES,
            core_ranges=core_set(cores),
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=ttnn.bfloat16, page_size=TILE_BYTES
                )
            ],
        )
        for i in (CB_IN, CB_OUT)
    ]
    return ttnn.ProgramDescriptor(kernels=[kernel, compute], semaphores=[], cbs=cbs)


def capture(args):
    """Builds the loader once in a fresh JIT cache with the commands logged (child
    process) and keeps the compile and link commands for the builder."""
    if args.child:
        with Session(card=args.card, log=log) as s:
            cores = [(0, 0)]
            ctl = l1_rows(s.device, cores, CTL_WORDS)
            slot = l1_rows(s.device, cores, SLOT_BYTES // 4)
            tslots = l1_rows(s.device, cores, 3 * TSLOT_BYTES // 4)
            results = dram_rows(s.device, np.zeros((1, RESP_WORDS)))
            module = native.load_module()
            built = module.kernel_build(
                s.device, loader_descriptor(cores, ctl, slot, tslots, results)
            )
            log(f"built: {built[0]}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS="1",
        TT_METAL_FORCE_JIT_COMPILE="1",
    )
    cmd = [sys.executable, __file__, "--card", str(args.card), "capture", "--child"]
    lines = []
    with subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    if child.returncode:
        raise SystemExit(f"capture child failed: {child.returncode}")
    kept = [ln for ln in lines if "g++ compile cmd:" in ln or "g++ link cmd:" in ln]
    COMMANDS.write_text("".join(kept))
    log(f"{len(kept)} build commands kept in {COMMANDS}")


class Rig:
    """An open card with the loader on `cores`: L1 control words and slots, a DRAM
    response page per core, the overlay images in DRAM (one page each)."""

    def __init__(self, session, cores, compute=False):
        self.s, self.dev, self.cores = session, session.device, cores
        self.module = native.load_module()
        self.ctl = l1_rows(self.dev, cores, CTL_WORDS)
        self.slot = l1_rows(self.dev, cores, SLOT_BYTES // 4)
        self.tslots = (
            l1_rows(self.dev, cores, 3 * TSLOT_BYTES // 4) if compute else None
        )
        self.results = dram_rows(self.dev, np.zeros((len(cores), RESP_WORDS)))
        self.desc = loader_descriptor(
            cores, self.ctl, self.slot, self.tslots, self.results
        )
        built = self.module.kernel_build(self.dev, self.desc)
        elves = {Path(src).name: [Path(e) for e in es if e] for src, es in built[0]}
        self.loader_elf = elves["loader.cpp"][0]
        self.trisc_elves = elves["compute.cpp"] if compute else []
        kernel, free, _ = self.module.kernel_buffer(self.dev)
        self.l1_base = int(kernel) + KERNEL_BUFFER_BASE
        text = COMMANDS.read_text()
        self.commands = builder.parse_commands(text, "loader", "brisc")
        self.tcommands = (
            [builder.parse_commands(text, "compute", t) for t in TRISCS]
            if compute
            else []
        )
        self.reserved = [
            ("firmware, mailboxes and the kernel buffer", 0, self.l1_base),
            (
                "the control words",
                self.ctl.buffer_address(),
                self.ctl.buffer_address() + 4 * CTL_WORDS,
            ),
        ]
        if self.tslots is not None:
            at = self.tslots.buffer_address()
            self.reserved.append(("the compute slots", at, at + 3 * TSLOT_BYTES))
        builder.check_slot(self.slot.buffer_address(), SLOT_BYTES, self.reserved)
        self.seq = [0] * len(cores)
        self.images = None
        log(
            f"loader {self.loader_elf.parent.parent.name}: slot "
            f"{self.slot.buffer_address():#x} + {SLOT_BYTES}, control "
            f"{self.ctl.buffer_address():#x}, L1 buffers from {self.l1_base:#x}"
        )

    def build(self, name, source, header=None):
        return builder.build(
            self.commands,
            name,
            source,
            header,
            self.slot.buffer_address(),
            SLOT_BYTES,
            self.loader_elf,
            OUT / "overlays",
        )

    def build_compute(self, name, source, header):
        """The three parts of a compute overlay, each linked for its core's slot
        against that core's persistent ELF, as one image (part k at k x TSLOT)."""
        base = self.tslots.buffer_address()
        parts = [
            builder.build(
                c,
                f"{name}-{t}",
                source,
                header,
                base + k * TSLOT_BYTES,
                TSLOT_BYTES,
                elf,
                OUT / "overlays",
                RESETS,
            )
            for k, (t, c, elf) in enumerate(
                zip(TRISCS, self.tcommands, self.trisc_elves)
            )
        ]
        image = b"".join(p.image + bytes(TSLOT_BYTES - len(p.image)) for p in parts[:2])
        image += parts[2].image
        return (
            builder.Overlay(name, image, base, sum(p.text for p in parts), None, None),
            parts,
        )

    def upload(self, overlays):
        """Puts every overlay's image in a DRAM page of its own; returns (id, bank,
        address, bytes) per overlay, ids from 1."""
        width = max(len(o.image) for o in overlays)
        rows = np.zeros((len(overlays), width // 4), np.uint32)
        for i, o in enumerate(overlays):
            rows[i, : len(o.image) // 4] = np.frombuffer(o.image, np.uint32)
        log(f"upload {len(overlays)} images of up to {width} B")
        self.images = dram_rows(self.dev, rows)
        return [
            (i + 1, *page_place(self.images, i), len(o.image))
            for i, o in enumerate(overlays)
        ]

    def launch(self):
        log("launch")
        ttnn.generic_op([self.results, self.ctl], self.desc)
        log("launched")

    def send(
        self, i, op, flags=0, arg=0, reps=0, a=(0,) * 4, b=(0,) * 4, expect=(0, 0)
    ):
        x, y = self.cores[i]
        self.seq[i] += 1
        words = [op, flags, arg & 0xFFFFFFFF, reps, *a, *b, *expect]
        at = self.ctl.buffer_address()
        self.module.write_l1(self.dev, x, y, at + 16, [int(w) for w in words])
        self.module.write_l1(self.dev, x, y, at, [self.seq[i]])
        return self.seq[i]

    def response(self, i, seq, timeout=5.0):
        """Core i's response line to `seq` (16 words), polled from DRAM."""
        by = time.monotonic() + timeout
        while True:
            (page,) = self.s.read(self.results, first=i, pages=1)
            words = page.reshape(-1).view(np.uint32)[:RESP_WORDS].astype(np.int64)
            if words[0] == seq:
                if words[1]:
                    raise RuntimeError(
                        f"core {i}: {STATUS.get(int(words[1]))}: {words}"
                    )
                return words
            if time.monotonic() > by:
                raise TimeoutError(f"core {i}: no response to {seq}: {words}")

    def broadcast(self, overlays, args, mhz, timeout=10.0):
        """OP_RUN of overlays[i] with argument args[i] on every core i, all
        starting at one wall-clock count LEAD_S after core 0's clock was read: the
        command words to all, then the sequence words in one pass. (Seconds of
        that pass, the start count, the response lines in core order.)"""
        at = self.ctl.buffer_address()
        now = int(self.call(0, OP_CLOCK)[2])
        start = (now + int(LEAD_S * mhz * 1e6)) % (1 << 32)
        for i, (x, y) in enumerate(self.cores):
            words = [
                OP_RUN,
                FLAG_INVALIDATE | FLAG_AT,
                args[i],
                start,
                *overlays[i],
            ]
            words += [0] * 6
            self.module.write_l1(self.dev, x, y, at + 16, [int(w) for w in words])
        t = time.perf_counter()
        for i, (x, y) in enumerate(self.cores):
            self.seq[i] += 1
            self.module.write_l1(self.dev, x, y, at, [self.seq[i]])
        host = time.perf_counter() - t
        by = time.monotonic() + timeout
        while True:
            (pages,) = self.s.read(self.results)
            words = pages.reshape(len(self.cores), -1).view(np.uint32)[:, :RESP_WORDS]
            if all(int(words[i, 0]) == self.seq[i] for i in range(len(self.cores))):
                bad = [i for i in range(len(self.cores)) if words[i, 1]]
                if bad:
                    raise RuntimeError(f"cores {bad[:8]}: status {words[bad[0]]}")
                return host, start, words.astype(np.int64)
            if time.monotonic() > by:
                raise TimeoutError("not every core responded")

    def call(self, i, *args, timeout=5.0, **kwargs):
        return self.response(i, self.send(i, *args, **kwargs), timeout)

    def exit(self):
        for i in range(len(self.cores)):
            self.send(i, OP_EXIT)
        for i in range(len(self.cores)):
            self.response(i, self.seq[i])
        g = self.dev.compute_with_storage_grid_size()
        done = self.module.wait_grid(self.dev, g.x, g.y, 10.0)
        log(f"loader left: {done}")


def chain(rig, name, steps, seed, last=None):
    """A chain overlay of `steps` steps and its host reference."""
    st = builder.chain_steps(steps, seed, last)
    o = rig.build(name, HERE / "chain.cpp", builder.chain_header(st))
    return o, (lambda arg: builder.chain_result(st, arg))


def steps_for(target):
    """Chain steps for an image of about `target` bytes (28 bytes a step, measured)."""
    return max(4, (target - 120) // 28)


def mhz_now(card):
    return aiclk_mhz(card)


def finish(job, out, suffix=""):
    out["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    text = json.dumps(out, default=lambda v: v.item())  # numpy scalars
    (OUT / f"{job}{suffix}.json").write_text(text)
    print(text, flush=True)


def stats(cycles, mhz):
    a = np.asarray(cycles, np.float64)
    return {
        "n": len(a),
        "median_cycles": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "median_us": round(float(np.median(a)) / mhz, 3),
    }


def swap(args):
    """Question 2: two overlays for one slot, swapped by command, results exact."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0)])
        (a, ref_a), (b, ref_b) = chain(rig, "swap-a", 120, 11), chain(
            rig, "swap-b", 140, 12
        )
        ids = rig.upload([a, b])
        mhz = mhz_now(args.card)
        rig.launch()
        rng = np.random.default_rng(5)
        wrong, loads, run_c, cmd_c, host = [], [], [], [], []
        calls = 0
        try:
            for k in range(args.reps):
                w = k % 2
                arg = int(rng.integers(0, 1 << 32))
                t = time.perf_counter()
                r = rig.call(0, OP_RUN, FLAG_INVALIDATE, arg, a=ids[w])
                host.append(time.perf_counter() - t)
                want = (ref_a, ref_b)[w](arg)
                calls += 1
                if int(r[2]) != want or int(r[15]) != calls or not r[4]:
                    wrong.append((k, int(r[2]), want, int(r[15]), int(r[4])))
                loads.append(int(r[5]))
                run_c.append(int(r[6]))
                cmd_c.append(int(r[7]))
                if k % 200 == 0:
                    log(f"swap {k}: result {int(r[2]):#x}, load {int(r[5])} cycles")
        finally:
            rig.exit()
    finish(
        "swap",
        {
            "swaps": args.reps,
            "wrong": len(wrong),
            "first_wrong": wrong[:5],
            "image_bytes": [len(a.image), len(b.image)],
            "aiclk_mhz": mhz,
            "load": stats(loads, mhz),
            "run": stats(run_c, mhz),
            "command": stats(cmd_c, mhz),
            "host_round_trip_ms": round(1e3 * float(np.median(host)), 3),
        },
    )


def cost(args):
    """Question 3: load time by body size, a command naming the overlay held."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0)])
        pairs = []
        for kib in args.sizes:
            n = steps_for(kib * 1024)
            pairs.append(
                (
                    kib,
                    chain(rig, f"cost{kib}a", n, 100 + kib),
                    chain(rig, f"cost{kib}b", n, 200 + kib),
                )
            )
        ids = rig.upload([o for _, a, b in pairs for o, _ in (a, b)])
        mhz = mhz_now(args.card)
        rig.launch()
        rows, rng = [], np.random.default_rng(7)
        try:
            for j, (kib, a, b) in enumerate(pairs):
                loads, runs, held, held_cmd, host_held = [], [], [], [], []
                bad = 0
                for k in range(args.reps):
                    w = k % 2
                    arg = int(rng.integers(0, 1 << 32))
                    r = rig.call(0, OP_RUN, FLAG_INVALIDATE, arg, a=ids[2 * j + w])
                    bad += int(r[2]) != (a, b)[w][1](arg) or not r[4]
                    loads.append(int(r[5]))
                    runs.append(int(r[6]))
                for _ in range(args.reps):
                    arg = int(rng.integers(0, 1 << 32))
                    t = time.perf_counter()
                    r = rig.call(0, OP_RUN, FLAG_INVALIDATE, arg, a=ids[2 * j + 1])
                    host_held.append(time.perf_counter() - t)
                    bad += int(r[2]) != b[1](arg) or r[4]
                    held.append(int(r[7]) - int(r[6]))
                    held_cmd.append(int(r[7]))
                size = len(a[0].image)
                rows.append(
                    {
                        "kib": kib,
                        "image_bytes": size,
                        "text_bytes": a[0].text,
                        "wrong": bad,
                        "load": stats(loads, mhz),
                        "run": stats(runs, mhz),
                        "held_command": stats(held_cmd, mhz),
                        "held_overhead": stats(held, mhz),
                        "host_round_trip_ms": round(
                            1e3 * float(np.median(host_held)), 3
                        ),
                        "bytes_per_cycle": round(size / float(np.median(loads)), 2),
                    }
                )
                med = (np.median(loads), np.median(runs))
                log(f"{kib} KiB: load {med[0]:.0f} cycles, run {med[1]:.0f}")
        finally:
            rig.exit()
    x = np.array([r["image_bytes"] for r in rows], np.float64) / 1024
    y = np.array([r["load"]["median_cycles"] for r in rows])
    slope, base = np.polyfit(x, y, 1)
    finish(
        "cost",
        {
            "aiclk_mhz": mhz,
            "sizes": rows,
            "fit_cycles": {
                "per_kib": round(float(slope), 1),
                "fixed": round(float(base), 1),
            },
        },
    )


def launch(args):
    """Question 3's comparison: the host's ordinary launch of a kernel on one core
    (slow dispatch), launch call to the core's return, with and without the
    runtime's program cache."""
    with Session(card=args.card, log=log) as s:
        dev = s.device
        dummy = dram_rows(dev, np.zeros((1, 16)))
        out = dram_rows(dev, np.zeros((1, 16)))
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(HERE / "empty.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_set([(0, 0)]),
            compile_time_args=[],
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default,
            ),
        )
        desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
        res = {}
        for cached in (False, True):
            if cached:
                dev.enable_program_cache()
            times = []
            for _ in range(args.reps + 1):
                t = time.perf_counter()
                ttnn.generic_op([dummy, out], desc)
                ttnn.synchronize_device(dev)
                times.append(time.perf_counter() - t)
            log(f"program cache {cached}: first {times[0] * 1e3:.2f} ms")
            res["program_cache" if cached else "no_program_cache"] = {
                "first_ms": round(times[0] * 1e3, 3),
                "median_ms": round(1e3 * float(np.median(times[1:])), 3),
                "min_ms": round(1e3 * float(np.min(times[1:])), 3),
                "n": args.reps,
            }
    finish("launch", res)


def stale(args):
    """Question 4: does a core run stale code after its slot is rewritten? Two bodies
    that differ in one late instruction (the last add's low immediate), alternated on
    the core, without and with the i-cache invalidated after each load."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0)])
        pairs = []
        for steps in args.steps:
            last = builder.chain_steps(steps, 300 + steps)[-1][2] ^ 1
            a = chain(rig, f"stale{steps}a", steps, 300 + steps)
            b = chain(rig, f"stale{steps}b", steps, 300 + steps, last)
            pairs.append((steps, a, b))
        ids = rig.upload([o for _, a, b in pairs for o, _ in (a, b)])
        mhz = mhz_now(args.card)
        rig.launch()
        rows = []
        try:
            for j, (steps, a, b) in enumerate(pairs):
                arg = 0x12345678 + j
                want = (a[1](arg), b[1](arg))
                for flags in (0, FLAG_INVALIDATE):
                    r = rig.call(
                        0,
                        OP_ALT,
                        flags,
                        arg,
                        args.reps,
                        ids[2 * j],
                        ids[2 * j + 1],
                        want,
                        timeout=60.0,
                    )
                    cycles = int(r[11]) | int(r[12]) << 32
                    rows.append(
                        {
                            "image_bytes": len(a[0].image),
                            "steps": steps,
                            "invalidate": bool(flags),
                            "alternations": args.reps,
                            "runs": int(r[10]),
                            "wrong": int(r[8]),
                            "stale": int(r[9]),
                            "last_wrong": f"{int(r[13]):#x}",
                            "cycles_per_load_and_run": round(
                                cycles / max(int(r[10]), 1), 1
                            ),
                        }
                    )
                    log(f"stale {rows[-1]}")
        finally:
            rig.exit()
    finish("stale", {"aiclk_mhz": mhz, "rows": rows})


def bf16_bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0]) >> 16


def scalar_result(add, scalar):
    """The host's reference of scalar.cpp through the loader: the sum of the output
    tile's bfloat16 bits for the input values (j % 16) + 1 (all exact)."""
    return 64 * sum(bf16_bits(v + scalar if add else v * scalar) for v in range(1, 17))


def compute(args):
    """Question 4, compute side: a compute overlay (unpack, math, pack parts) loaded
    by the data-movement core while the compute cores are parked, swapped with a
    second one, every result exact."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0)], compute=True)
        a, parts_a = rig.build_compute("mul", HERE / "scalar.cpp", "#define OV_ADD 0\n")
        b, parts_b = rig.build_compute("add", HERE / "scalar.cpp", "#define OV_ADD 1\n")
        sizes = {
            o.name: [len(p.image) for p in parts]
            for o, parts in ((a, parts_a), (b, parts_b))
        }
        log(f"compute parts: {sizes} B (unpack, math, pack)")
        ids = rig.upload([a, b])
        mhz = mhz_now(args.card)
        rig.launch()
        wrong, loads, runs = [], [], []
        try:
            for k in range(args.reps):
                w = k % 2
                scalar = float(2 + k % 7)
                arg = int(np.array([scalar], np.float32).view(np.uint32)[0])
                r = rig.call(0, OP_COMPUTE, FLAG_INVALIDATE, arg, a=ids[w])
                want = scalar_result(w == 1, scalar)
                if int(r[2]) != want or not r[4]:
                    wrong.append((k, int(r[2]), want, int(r[4])))
                loads.append(int(r[5]))
                runs.append(int(r[6]))
                if k % 200 == 0 or k < 2:
                    log(f"compute {k}: sum {int(r[2])} (want {want}), load {int(r[5])}")
        finally:
            rig.exit()
    finish(
        "compute",
        {
            "swaps": args.reps,
            "wrong": len(wrong),
            "first_wrong": wrong[:5],
            "part_bytes": sizes,
            "image_bytes": [len(a.image), len(b.image)],
            "aiclk_mhz": mhz,
            "load": stats(loads, mhz),
            "run": stats(runs, mhz),
        },
    )


def grid(args):
    """Question 5: every core loads the same overlay from DRAM on one command (one
    copy, one DRAM bank); then a command naming the overlay already held."""
    with Session(card=args.card, log=log) as s:
        g = s.device.compute_with_storage_grid_size()
        cores = [(x, y) for y in range(g.y) for x in range(g.x)]
        rig = Rig(s, cores)
        n = steps_for(args.kib * 1024)
        a, b = chain(rig, "grid-a", n, 501), chain(rig, "grid-b", n, 502)
        # copies of A, then of B: one DRAM page each, page i in bank i % 8; core i
        # reads copy i % copies, under the overlay's one id
        ids = rig.upload([a[0]] * args.copies + [b[0]] * args.copies)
        mhz = mhz_now(args.card)
        rig.launch()
        rounds = []
        try:
            for k in range(args.rounds):
                for held in (False, True):
                    w = k % 2
                    copies = ids[w * args.copies : (w + 1) * args.copies]
                    host, start, resp = rig.broadcast(
                        [
                            (w + 1, *copies[i % args.copies][1:])
                            for i in range(len(cores))
                        ],
                        [0x1000 + 7 * i + k for i in range(len(cores))],
                        mhz,
                    )
                    ref = (a, b)[w][1]
                    wrong = sum(
                        int(r[2]) != ref(0x1000 + 7 * i + k) for i, r in enumerate(resp)
                    )
                    began = np.array([int(r[14]) for r in resp], np.int64)
                    seen = np.array([int(r[13]) for r in resp], np.int64)
                    done = (began + np.array([int(r[7]) for r in resp])) % (1 << 32)
                    late = int(sum(((seen - start) % (1 << 32)) < (1 << 31)))
                    loaded = sum(int(r[4]) for r in resp)
                    rounds.append(
                        {
                            "overlay": "ab"[w],
                            "held": held,
                            "wrong": wrong,
                            "loaded": loaded,
                            "start_to_last_done_us": round(
                                float(((done - start) % (1 << 32)).max()) / mhz, 3
                            ),
                            "start_spread_us": round(
                                float(((began - start) % (1 << 32)).max()) / mhz, 3
                            ),
                            "seen_after_start": late,
                            "load_per_core": (
                                stats([int(r[5]) for r in resp], mhz)
                                if not held
                                else None
                            ),
                            "command_per_core": stats([int(r[7]) for r in resp], mhz),
                            "host_issue_ms": round(1e3 * host, 3),
                        }
                    )
                    log(f"grid round {k} held {held}: {rounds[-1]}")
        finally:
            rig.exit()
    finish(
        "grid",
        {
            "cores": len(cores),
            "image_bytes": len(a[0].image),
            "aiclk_mhz": mhz,
            "dram_copies": args.copies,
            "rounds": rounds,
        },
        f"-{args.copies}",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--card", type=int, default=1)
    sub = p.add_subparsers(dest="job", required=True)
    c = sub.add_parser("capture", help="the loader's build commands")
    c.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("swap", help="question 2: two overlays swapped")
    c.add_argument("--reps", type=int, default=1000)
    c = sub.add_parser("cost", help="question 3: load time by size")
    c.add_argument("--sizes", type=int, nargs="+", default=[4, 10, 20, 40, 80])
    c.add_argument("--reps", type=int, default=20)
    c = sub.add_parser("launch", help="question 3: the host's launch of one core")
    c.add_argument("--reps", type=int, default=20)
    c = sub.add_parser("stale", help="question 4: stale code after a rewrite")
    c.add_argument("--steps", type=int, nargs="+", default=[16, 120])
    c.add_argument("--reps", type=int, default=10000)
    c = sub.add_parser("compute", help="question 4: a compute overlay swapped")
    c.add_argument("--reps", type=int, default=1000)
    c = sub.add_parser("grid", help="question 5: every core, one command")
    c.add_argument("--kib", type=int, default=20)
    c.add_argument("--rounds", type=int, default=4)
    c.add_argument("--copies", type=int, default=1, help="copies in DRAM (banks)")
    args = p.parse_args()
    jobs = {
        "grid": grid,
        "compute": compute,
        "stale": stale,
        "capture": capture,
        "swap": swap,
        "cost": cost,
        "launch": launch,
    }
    jobs[args.job](args)


if __name__ == "__main__":
    main()
