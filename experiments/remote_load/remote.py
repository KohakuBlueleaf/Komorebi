"""Remote loading: a Tensix core stores bodies into other cores' slots; an X280 thread
decides each core's next body (the core reads it), stores it, or patches a template.
Progress lines, then one JSON line. Example: python remote.py capture; ... swap"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / "experiments" / "overlay_probe"
OUT = ROOT / "build" / "remote_load"
# One JIT cache for every job: the kernel's ELF the bodies link against is the one
# that runs. Set before tt-metal is loaded.
os.environ["TT_METAL_CACHE"] = str(OUT / "jit-cache")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROBE))

import builder  # noqa: E402
import numpy as np  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import native  # noqa: E402
from komorebi.session import KERNEL_BUFFER_BASE, Session, aiclk_mhz  # noqa: E402

T0 = time.monotonic()
CTL_WORDS = 128  # remote.cpp: control words per core
RESP_WORDS = 16
MSG, REPLY = 64, 68  # remote.cpp: the message words, the reply address words
OP_FETCH, OP_FEED, OP_EXIT, OP_CLOCK, OP_STALE = 1, 2, 3, 5, 6
FLAG_TABLE, FLAG_REMOTE = 1, 2
STATUS = {
    0: "ok",
    1: "bad op",
    2: "read timeout",
    3: "too big",
    4: "idle",
    5: "target not parked",
    6: "no reply",
    7: "write timeout",
}
SLOT_BYTES = 96 * 1024
TARGETS_MAX = 128
TABLE_WORDS = 16 * TARGETS_MAX + TARGETS_MAX + 4 * TARGETS_MAX
TABLE_TARGETS = 16 * TARGETS_MAX
IDLE_UNITS = 2000  # a core leaves after this many 2^20 cycles without a command
COMMANDS = OUT / "build-commands.txt"


def log(msg):
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def core_set(cores):
    return [(x, y, x, y) for x, y in cores]


def l1_rows(session, cores, words):
    """A zeroed uint32 L1 tensor with one row of `words` words on each core: `cores`
    are a grid from (0, 0) in row-major order."""
    gx, gy = max(x for x, _ in cores) + 1, max(y for _, y in cores) + 1
    if cores != [(x, y) for y in range(gy) for x in range(gx)]:
        raise ValueError("the cores of an L1 tensor are a grid from (0, 0)")
    rows = np.zeros((len(cores), words), np.uint32)
    return session.tensor(rows, "uint32", shard_grid=(gx, gy))


def dram_rows(session, rows):
    """A uint32 DRAM tensor holding the rows (one page a row)."""
    return session.tensor(np.asarray(rows, np.uint32), "uint32")


def page_place(tensor, page):
    """(bank, address) of an interleaved DRAM tensor's page."""
    return page % 8, tensor.address + (page // 8) * tensor.aligned_page_bytes


class Rig:
    """An open card with remote.cpp on `cores`: per core L1 control words, a slot, a
    store of two bodies and a table; a DRAM response page per core and one table."""

    def __init__(self, session, cores):
        self.s, self.dev, self.cores = session, session.device, cores
        self.module = native.load_module()
        self.ctl = l1_rows(session, cores, CTL_WORDS)
        self.slot = l1_rows(session, cores, SLOT_BYTES // 4)
        self.store = l1_rows(session, cores, 2 * SLOT_BYTES // 4)
        self.table = l1_rows(session, cores, TABLE_WORDS)
        self.results = dram_rows(session, np.zeros((len(cores), RESP_WORDS)))
        self.dram_table = dram_rows(session, np.zeros((1, 16 * TARGETS_MAX)))
        self.nocs = session.nocs().noc_xy
        self.kernel = self.launcher()
        built = self.kernel.build()
        elves = {Path(src).name: [Path(e) for e in es if e] for src, es in built[0]}
        self.elf = elves["remote.cpp"][0]
        kernel, _, _ = self.module.kernel_buffer(self.dev)
        self.l1_base = int(kernel) + KERNEL_BUFFER_BASE
        self.commands = builder.parse_commands(COMMANDS.read_text(), "remote", "brisc")
        ctl = self.ctl.address
        reserved = [
            ("firmware, mailboxes and the kernel buffer", 0, self.l1_base),
            ("the control words", ctl, ctl + 4 * CTL_WORDS),
        ]
        for name, buf, words in (
            ("the store", self.store, 2 * SLOT_BYTES // 4),
            ("the table", self.table, TABLE_WORDS),
        ):
            at = buf.address
            reserved.append((name, at, at + 4 * words))
        builder.check_slot(self.slot.address, SLOT_BYTES, reserved)
        self.seq = [0] * len(cores)
        log(
            f"remote {self.elf.parent.parent.name}: slot "
            f"{self.slot.address:#x} + {SLOT_BYTES}, store "
            f"{self.store.address:#x}, table {self.table.address:#x}"
            f", control {ctl:#x}, L1 buffers from {self.l1_base:#x}"
        )

    def launcher(self):
        """remote.cpp on the rig's cores, with each core's arguments."""
        args = []
        tbank, tat = page_place(self.dram_table, 0)
        for i, (x, y) in enumerate(self.cores):
            bank, at = page_place(self.results, i)
            words = [self.ctl.address, bank, at, self.slot.address, SLOT_BYTES]
            words += [IDLE_UNITS, self.store.address, self.table.address, tbank, tat]
            args.append((x, y, words))
        launch = self.s.launch()
        launch.kernel(
            str(HERE / "remote.cpp"), core_set(self.cores), 0, [], [], [str(PROBE)],
            args,
        )  # fmt: skip
        return launch

    def xy(self, i):
        """Core i's translated NoC coordinates as one word, x | y << 16."""
        x, y = self.nocs[self.cores[i]]
        return x | y << 16

    def build(self, name, steps, seed):
        """A chain body (overlay_probe/chain.cpp) for the slot and its reference."""
        st = builder.chain_steps(steps, seed)
        o = builder.build(
            self.commands,
            name,
            PROBE / "chain.cpp",
            builder.chain_header(st),
            self.slot.address,
            SLOT_BYTES,
            self.elf,
            OUT / "bodies",
        )
        return o, (lambda arg: builder.chain_result(st, arg))

    def upload(self, overlays):
        """Every body's image in a DRAM page of its own; (id, bank, address, bytes)
        per body, ids from 1."""
        width = max(len(o.image) for o in overlays)
        rows = np.zeros((len(overlays), width // 4), np.uint32)
        for i, o in enumerate(overlays):
            rows[i, : len(o.image) // 4] = np.frombuffer(o.image, np.uint32)
        log(f"upload {len(overlays)} images of up to {width} B")
        self.images = dram_rows(self.s, rows)
        return [
            (i + 1, *page_place(self.images, i), len(o.image))
            for i, o in enumerate(overlays)
        ]

    def wire(self, writer, targets):
        """Writer core index and target core indices: the writer's target words, each
        target's reply address (its line in the writer's table). Before the launch."""
        table = self.table.address
        x, y = self.cores[writer]
        words = [self.xy(i) for i in targets]
        self.module.write_l1(self.dev, x, y, table + 4 * TABLE_TARGETS, words)
        for k, i in enumerate(targets):
            tx, ty = self.cores[i]
            at = self.ctl.address + 4 * REPLY
            self.module.write_l1(
                self.dev, tx, ty, at, [self.xy(writer), table + 64 * k]
            )

    def launch(self):
        log("launch")
        self.kernel.run()
        log("launched")

    def send(
        self, i, op, flags=0, arg=0, count=0, a=(0,) * 4, b=(0,) * 4, which=0,
        expect=(0, 0),
    ):  # fmt: skip
        x, y = self.cores[i]
        self.seq[i] += 1
        words = [op, flags, arg & 0xFFFFFFFF, count, *a, *b, which, 0, *expect]
        at = self.ctl.address
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

    def call(self, i, *args, timeout=5.0, **kwargs):
        return self.response(i, self.send(i, *args, **kwargs), timeout)

    def lines(self, n):
        """The writer's reply lines of targets 0..n-1 (after a feed with the table)."""
        (page,) = self.s.read(self.dram_table)
        return (
            page.reshape(-1).view(np.uint32)[: 16 * n].reshape(n, 16).astype(np.int64)
        )

    def exit(self):
        for i in range(len(self.cores)):
            self.send(i, OP_EXIT)
        for i in range(len(self.cores)):
            self.response(i, self.seq[i])
        gx, gy = self.s.grid
        done = self.module.wait_grid(self.dev, gx, gy, 10.0)
        log(f"kernel left: {done}")


def steps_for(target):
    """Chain steps for an image of about `target` bytes (28 bytes a step, measured)."""
    return max(4, (target - 120) // 28)


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


def capture(args):
    """Builds remote.cpp once in a fresh JIT cache with the commands logged (child
    process) and keeps the compile and link commands for the builder."""
    if args.child:
        with Session(card=args.card, log=log) as s:
            cores = [(0, 0)]
            launch = s.launch()
            launch.kernel(
                str(HERE / "remote.cpp"), core_set(cores), 0, [], [], [str(PROBE)], []
            )
            built = launch.build()
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


def feed(rig, writer, n, w, arg, table=False, timeout=5.0):
    return rig.call(
        writer, OP_FEED, FLAG_TABLE if table else 0, arg, n, which=w, timeout=timeout
    )


def swap(args):
    """Writer core 0 swaps two bodies in target core 1's slot, 1,000 times;
    every result, the target's call counter and the body's id checked."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0), (1, 0)])
        (a, ref_a), (b, ref_b) = rig.build("swap-a", 120, 11), rig.build(
            "swap-b", 140, 12
        )
        ids = rig.upload([a, b])
        rig.wire(0, [1])
        mhz = aiclk_mhz(args.card)
        rig.launch()
        rows, wrong = [], []
        rng = np.random.default_rng(5)
        try:
            r = rig.call(0, OP_FETCH, a=ids[0], b=ids[1])
            log(f"fetch: {int(r[5])} cycles")
            for k in range(args.reps):
                w = k % 2
                arg = int(rng.integers(0, 1 << 32))
                r = feed(rig, 0, 1, w, arg)
                want = (ref_a, ref_b)[w](arg)
                if int(r[2]) != want or int(r[4]) != k + 1 or int(r[9]) != w + 1:
                    wrong.append((k, int(r[2]), want, int(r[4]), int(r[9])))
                rows.append((int(r[5]), int(r[7]), int(r[6]), int(r[8])))
                if k % 200 == 0:
                    log(f"swap {k}: result {int(r[2]):#x}, store {int(r[5])} cycles")
        finally:
            rig.exit()
    st, cmd, run, sent = np.array(rows).T
    finish(
        "swap",
        {
            "swaps": args.reps,
            "wrong": len(wrong),
            "first_wrong": wrong[:5],
            "image_bytes": [len(a.image), len(b.image)],
            "aiclk_mhz": mhz,
            "store": stats(st, mhz),
            "command_to_reply": stats(cmd, mhz),
            "target_run": stats(run, mhz),
            "messages_sent": stats(sent, mhz),
            "command_minus_store_and_run": stats(cmd - st - run, mhz),
        },
    )


def sizes(args):
    """The writer's store of 4, 20 and 80 KiB into one target, and the command
    to the target's reply; two bodies a size alternated, every result checked."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0), (1, 0)])
        pairs = []
        for kib in args.sizes:
            n = steps_for(kib * 1024)
            pairs.append(
                (
                    kib,
                    rig.build(f"s{kib}a", n, 100 + kib),
                    rig.build(f"s{kib}b", n, 200 + kib),
                )
            )
        ids = rig.upload([o for _, a, b in pairs for o, _ in (a, b)])
        rig.wire(0, [1])
        mhz = aiclk_mhz(args.card)
        rig.launch()
        out, calls, rng = [], 0, np.random.default_rng(7)
        try:
            for j, (kib, a, b) in enumerate(pairs):
                rig.call(0, OP_FETCH, a=ids[2 * j], b=ids[2 * j + 1])
                st, cmd, run, bad = [], [], [], 0
                for k in range(args.reps):
                    w = k % 2
                    arg = int(rng.integers(0, 1 << 32))
                    r = feed(rig, 0, 1, w, arg)
                    calls += 1
                    bad += int(r[2]) != (a, b)[w][1](arg) or int(r[4]) != calls
                    st.append(int(r[5]))
                    cmd.append(int(r[7]))
                    run.append(int(r[6]))
                size = len(a[0].image)
                out.append(
                    {
                        "kib": kib,
                        "image_bytes": size,
                        "wrong": bad,
                        "store": stats(st, mhz),
                        "bytes_per_cycle": round(size / float(np.median(st)), 2),
                        "command_to_reply": stats(cmd, mhz),
                        "target_run": stats(run, mhz),
                    }
                )
                med = (np.median(st), np.median(cmd))
                log(f"{kib} KiB: store {med[0]:.0f}, command {med[1]:.0f}")
        finally:
            rig.exit()
    x = np.array([r["image_bytes"] for r in out], np.float64) / 1024
    y = np.array([r["store"]["median_cycles"] for r in out])
    slope, base = np.polyfit(x, y, 1)
    finish(
        "sizes",
        {
            "aiclk_mhz": mhz,
            "sizes": out,
            "store_fit_cycles": {
                "per_kib": round(float(slope), 1),
                "fixed": round(float(base), 1),
            },
        },
    )


def fan(args):
    """One writer (core 0) feeds `--targets` cores the same 20 KiB body, rounds
    alternating two bodies; every target's result and counter checked."""
    with Session(card=args.card, log=log) as s:
        gx, gy = s.grid
        cores = [(x, y) for y in range(gy) for x in range(gx)][: args.targets + 1]
        rig = Rig(s, cores)
        n = steps_for(args.kib * 1024)
        a, b = rig.build("fan-a", n, 501), rig.build("fan-b", n, 502)
        ids = rig.upload([a[0], b[0]])
        targets = list(range(1, len(cores)))
        rig.wire(0, targets)
        mhz = aiclk_mhz(args.card)
        rig.launch()
        rounds = []
        try:
            rig.call(0, OP_FETCH, a=ids[0], b=ids[1])
            for k in range(args.rounds):
                w = k % 2
                base = 0x1000 * (k + 1)
                r = feed(rig, 0, len(targets), w, base, table=True, timeout=10.0)
                lines = rig.lines(len(targets))
                ref = (a, b)[w][1]
                wrong = sum(
                    int(lines[i, 0]) != ref(base + i)
                    or int(lines[i, 1]) != k + 1
                    or int(lines[i, 6]) != w + 1
                    for i in range(len(targets))
                )
                seen = lines[:, 3]
                done = lines[:, 4]
                rounds.append(
                    {
                        "body": "ab"[w],
                        "wrong": wrong,
                        "store_us": round(int(r[5]) / mhz, 3),
                        "messages_sent_us": round(int(r[8]) / mhz, 3),
                        "command_to_last_reply_us": round(int(r[7]) / mhz, 3),
                        "target_run": stats(lines[:, 2], mhz),
                        "message_seen_spread_us": round(
                            float((seen - seen.min()).max()) / mhz, 3
                        ),
                        "done_spread_us": round(
                            float((done - done.min()).max()) / mhz, 3
                        ),
                    }
                )
                log(f"fan round {k}: {rounds[-1]}")
        finally:
            rig.exit()
    finish(
        "fan",
        {
            "targets": len(targets),
            "image_bytes": len(a[0].image),
            "aiclk_mhz": mhz,
            "rounds": rounds,
        },
        f"-{len(targets)}",
    )


def stale(args):
    """Writer core 0 alternates two bodies that differ in one late immediate
    through target core 1's slot; the target does not invalidate its i-cache. Without,
    then with the writer's write of the target's invalidate register through the NoC."""
    with Session(card=args.card, log=log) as s:
        rig = Rig(s, [(0, 0), (1, 0)])
        pairs = []
        for steps in args.steps:
            last = builder.chain_steps(steps, 300 + steps)[-1][2] ^ 1
            st_a = builder.chain_steps(steps, 300 + steps)
            st_b = builder.chain_steps(steps, 300 + steps, last)
            a = builder.build(
                rig.commands, f"stale{steps}a", PROBE / "chain.cpp",
                builder.chain_header(st_a), rig.slot.address, SLOT_BYTES,
                rig.elf, OUT / "bodies",
            )  # fmt: skip
            b = builder.build(
                rig.commands, f"stale{steps}b", PROBE / "chain.cpp",
                builder.chain_header(st_b), rig.slot.address, SLOT_BYTES,
                rig.elf, OUT / "bodies",
            )  # fmt: skip
            pairs.append((steps, a, b, st_a, st_b))
        ids = rig.upload([o for _, a, b, _, _ in pairs for o in (a, b)])
        rig.wire(0, [1])
        mhz = aiclk_mhz(args.card)
        rig.launch()
        rows = []
        try:
            for j, (steps, a, _, st_a, st_b) in enumerate(pairs):
                rig.call(0, OP_FETCH, a=ids[2 * j], b=ids[2 * j + 1])
                arg = 0x12345678 + j
                want = (
                    builder.chain_result(st_a, arg),
                    builder.chain_result(st_b, arg),
                )
                for remote in args.remote:
                    log(f"stale {steps} steps, remote {remote}: {args.reps} times")
                    r = rig.call(
                        0, OP_STALE, FLAG_REMOTE if remote else 0, arg, args.reps,
                        expect=want, timeout=60.0,
                    )  # fmt: skip
                    cycles = int(r[11]) | int(r[12]) << 32
                    rows.append(
                        {
                            "image_bytes": len(a.image),
                            "steps": steps,
                            "writer": "tensix",
                            "remote_invalidate": bool(remote),
                            "alternations": args.reps,
                            "runs": int(r[10]),
                            "wrong": int(r[8]),
                            "stale": int(r[9]),
                            "last_wrong": f"{int(r[13]):#x}",
                            "cycles_per_store_and_run": round(
                                cycles / max(int(r[10]), 1), 1
                            ),
                        }
                    )
                    log(f"stale {rows[-1]}")
        finally:
            rig.exit()
    finish(
        "stale", {"aiclk_mhz": mhz, "rows": rows}, "-" + "-".join(map(str, args.remote))
    )


# remote_x280.h
RL_LINE, RL_LINES_MAX, RL_SCHED, RL_BODIES_MAX = 8, 12800, 8, 8
OUT_BYTES = 4 * RL_LINE * RL_LINES_MAX
MODES = {"self": 1, "store": 2, "patch": 3}
RESULT, CALLS, CYCLES, TICKS, STORE, BODY, ARG, WHO = range(8)
TICKS_PER_US = 50  # kmx_time() counts the 50 MHz timer
MARK = 0x2468ACE7  # the template's constant as built: bit 11 set (addi negative)
M32 = 0xFFFFFFFF


def rl_arg(c, k):
    return ((c << 20) ^ (k * 0x9E3779B1)) & M32


def rl_value(k):
    return (k * 0x9E3779B9 + 0x1234) & M32


def body_result(steps, arg, constant=None):
    """The host's reference of body.cpp: the chain, plus the constant, then mix()."""
    x = arg ^ 1
    for k, r, k2 in steps:
        y = x ^ k
        x = ((((y << r) | (y >> (32 - r))) & M32) + k2) & M32
    if constant is not None:
        x = (x + constant) & M32
    return builder.mix(x)


def lui_addi(rd, value):
    """The lui / addi pair that loads `value` into register rd (fields masked)."""
    hi, lo = ((value + 0x800) >> 12) & 0xFFFFF, value & 0xFFF
    return hi << 12 | rd << 7 | 0x37, lo << 20 | rd << 15 | rd << 7 | 0x13


def find_mark(image, value):
    """(byte offset, rd) of the one lui / addi pair of `image` that loads `value`;
    its encoding is asserted against lui_addi()."""
    words = np.frombuffer(image, np.uint32)
    found = []
    for i in range(len(words) - 1):
        a, b = int(words[i]), int(words[i + 1])
        rd = (a >> 7) & 0x1F
        if a & 0x7F != 0x37 or b & 0x707F != 0x13:
            continue
        if (b >> 7) & 0x1F != rd or (b >> 15) & 0x1F != rd:
            continue
        lo = b >> 20
        lo -= (lo & 0x800) << 1
        if ((a & 0xFFFFF000) + lo) & M32 == value:
            found.append((4 * i, rd, a, b))
    if len(found) != 1:
        raise builder.OverlayError(f"{len(found)} lui / addi pairs load {value:#x}")
    at, rd, a, b = found[0]
    if (a, b) != lui_addi(rd, value):
        raise builder.OverlayError(f"pair at {at}: {a:#x} {b:#x} not as encoded")
    return at, rd


def remote_elf():
    """The ELF of remote.cpp in the JIT cache (capture): the bodies link against it
    (they name none of its symbols, so they run under any persistent kernel)."""
    elves = sorted(
        (OUT / "jit-cache").glob("*/kernels/remote/*/brisc/brisc.elf"),
        key=lambda p: p.stat().st_mtime,
    )
    if not elves:
        raise SystemExit("no remote.cpp ELF: run capture first")
    return elves[-1]


def build_body(slot, name, steps, seed, mark=False):
    """A body.cpp image for the slot, its steps, and (mark) the constant's place."""
    st = builder.chain_steps(steps, seed)
    header = builder.chain_header(st)
    header += (
        f"// body.h {hashlib.sha256((HERE / 'body.h').read_bytes()).hexdigest()}\n"
    )
    if mark:
        header += f"#define OV_MARK {MARK:#x}\n"
    commands = builder.parse_commands(COMMANDS.read_text(), "remote", "brisc")
    o = builder.build(
        commands, name, HERE / "body.cpp", header, slot, SLOT_BYTES, remote_elf(),
        OUT / "bodies",
    )  # fmt: skip
    return o, st, find_mark(o.image, MARK) if mark else None


def x280_job(args):
    """An X280 thread decides every swap of `--cores` Tensix cores from its
    X280 cores' schedules; by `--mode` the core reads the body from DRAM (self), or
    the X280 core stores it (store) or patches the template's constant (patch)."""
    mode = MODES[args.mode]
    with km.Session(card=args.card, log=log) as s:
        gx, gy = s.grid
        grid = [(x, y) for y in range(gy) for x in range(gx)]
        slot_t = l1_rows(s, grid, SLOT_BYTES // 4)
        slot = slot_t.address
        kernel, _ = s.l1()
        builder.check_slot(
            slot,
            SLOT_BYTES,
            [("firmware and the kernel buffer", 0, KERNEL_BUFFER_BASE + kernel)],
        )
        sizes = args.sizes if mode != MODES["patch"] else [args.sizes[0]]
        bodies, mark = [], None  # (overlay, steps), two a size (one: the template)
        for kib in sizes:
            n = steps_for(kib * 1024) if kib else 120
            if mode == MODES["patch"]:
                o, st, mark = build_body(slot, f"tpl{kib}", n, 900 + kib, mark=True)
                bodies.append((o, st))
            else:
                for j in range(2):
                    o, st, _ = build_body(
                        slot, f"b{kib}{'ab'[j]}", n + 20 * j, 700 + 10 * kib + j
                    )
                    bodies.append((o, st))
        log(
            f"slot {slot:#x}; bodies {[len(o.image) for o, _ in bodies]} B; mark {mark}"
        )
        imgs = [o.image for o, _ in bodies]
        width = max(len(i) for i in imgs)
        rows = np.zeros((8 * len(imgs), width // 4), np.uint32)
        for b, image in enumerate(imgs):
            rows[8 * b : 8 * b + 8, : len(image) // 4] = np.frombuffer(image, np.uint32)
        images = dram_rows(s, rows)
        offsets = np.cumsum([0] + [len(i) for i in imgs])[:-1]
        region = np.zeros(int(offsets[-1]) + len(imgs[-1]), np.uint8)
        for at, image in zip(offsets, imgs):
            region[at : at + len(image)] = np.frombuffer(image, np.uint8)
        nocs = s.nocs().noc_xy
        per = 1 if mode == MODES["patch"] else 2
        rng = np.random.default_rng(13)
        sched = rng.integers(0, per, (16, RL_SCHED))
        cfg = np.zeros(2 * RL_BODIES_MAX + 16 * RL_SCHED + 128, np.uint32)
        cfg[: len(imgs)] = [len(i) for i in imgs]
        cfg[RL_BODIES_MAX : RL_BODIES_MAX + len(imgs)] = offsets
        cfg[2 * RL_BODIES_MAX : 2 * RL_BODIES_MAX + 16 * RL_SCHED] = sched.reshape(-1)
        noc_words = [nocs[c][0] | nocs[c][1] << 16 for c in grid]
        cfg[2 * RL_BODIES_MAX + 16 * RL_SCHED :][: len(grid)] = noc_words
        cores = len(grid) if args.cores == 0 else args.cores
        clusters = args.clusters
        out = {"mode": args.mode, "cores": cores, "clusters": clusters, "runs": []}
        for j, kib in enumerate(sizes):
            first = j * per
            prog = km.Program(
                "remote_load",
                kernels=HERE / "kernels",
                args=lambda core: ([slot, SLOT_BYTES], [], []),
            )
            prog.thread(
                clusters=clusters,
                sources=[HERE / "thread" / "remote.c"],
                includes=[HERE],
                regions={"cfg": cfg, "bodies": region, "out": OUT_BYTES},
                words=[
                    mode, args.swaps, cores, slot, images.address,
                    images.aligned_page_bytes, first,
                    mark[0] if mark else 0, mark[1] if mark else 0,
                ],
                deadline=args.deadline,
            )  # fmt: skip
            if cores == 1:
                prog.ports({0: (clusters[0], 0)})
            else:
                prog.ports()
            run = s.load(prog)
            mhz = aiclk_mhz(args.card)
            log(f"loaded {args.mode} {kib} KiB, aiclk {mhz} MHz")
            try:
                results = run.thread_results(timeout=args.deadline + 10)
            except km.ThreadFailed as e:
                results = e.results
            for c, r in results.items():
                log(
                    f"cluster {c}: {r.state}, code {r.code}, error {r.error}, {r.words}"
                )
            run.stop()
            lines = np.zeros((cores * args.swaps, RL_LINE), np.int64)
            for c, t in run.threads.items():
                got = t.read("out").view(np.uint32)[: cores * args.swaps * RL_LINE]
                got = got.reshape(-1, RL_LINE).astype(np.int64)
                mine = (got[:, WHO] != 0) & ((got[:, WHO] - 1) >> 2 == c)
                lines[mine] = got[mine]
            out["runs"].append(
                check(
                    lines,
                    cores,
                    args.swaps,
                    bodies,
                    sched,
                    first,
                    mode,
                    mhz,
                    kib,
                    results,
                )
            )
            log(f"run: {out['runs'][-1]}")
    finish(f"x280-{args.mode}", out, f"-{cores}-" + "-".join(map(str, sizes)))


def check(lines, cores, swaps, bodies, sched, first, mode, mhz, kib, results):
    """Every line against the host's reference: body by the schedule of the X280 core
    that served it, the argument, the call counter, the result."""
    wrong, missing = 0, 0
    for c in range(cores):
        for k in range(swaps):
            ln = lines[c * swaps + k]
            if ln[WHO] == 0:
                missing += 1
                continue
            who = int(ln[WHO]) - 1
            b = first + int(sched[who, k % RL_SCHED])
            o, st = bodies[b]
            arg = rl_arg(c, k)
            want = body_result(st, arg, rl_value(k) if mode == MODES["patch"] else None)
            if (
                int(ln[BODY]) != b
                or int(ln[ARG]) != arg
                or int(ln[CALLS]) != k + 1
                or int(ln[RESULT]) != want
            ):
                wrong += 1
    ticks = lines[:, TICKS]
    lead = results[min(results)]
    w = list(lead.words) + [0] * 4
    return {
        "kib": kib,
        "image_bytes": len(bodies[first][0].image),
        "swaps": cores * swaps,
        "wrong": wrong,
        "missing": missing,
        "codes": {c: r.code for c, r in results.items()},
        "aiclk_mhz": mhz,
        "swap_us_x280": {
            "median": round(float(np.median(ticks)) / TICKS_PER_US, 3),
            "min": round(float(ticks.min()) / TICKS_PER_US, 3),
            "max": round(float(ticks.max()) / TICKS_PER_US, 3),
        },
        "store_us_x280": {
            "median": round(float(np.median(lines[:, STORE])) / TICKS_PER_US, 3),
            "min": round(float(lines[:, STORE].min()) / TICKS_PER_US, 3),
            "max": round(float(lines[:, STORE].max()) / TICKS_PER_US, 3),
        },
        "load_cycles": stats(lines[:, CYCLES] >> 16, mhz),
        "run_cycles": stats(lines[:, CYCLES] & 0xFFFF, mhz),
        "command_us": round(((w[3] << 32) + w[2]) / TICKS_PER_US, 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--card", type=int, default=1)
    sub = p.add_subparsers(dest="job", required=True)
    c = sub.add_parser("capture", help="the kernel's build commands")
    c.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("swap", help="core to core: two bodies swapped")
    c.add_argument("--reps", type=int, default=1000)
    c = sub.add_parser("sizes", help="core to core: the store by size")
    c.add_argument("--sizes", type=int, nargs="+", default=[4, 20, 80])
    c.add_argument("--reps", type=int, default=20)
    c = sub.add_parser("fan", help="core to core: one writer, many targets")
    c.add_argument("--targets", type=int, default=8)
    c.add_argument("--kib", type=int, default=20)
    c.add_argument("--rounds", type=int, default=10)
    c = sub.add_parser("stale", help="the writer invalidates the target's i-cache")
    c.add_argument("--steps", type=int, nargs="+", default=[16, 120])
    c.add_argument("--reps", type=int, default=10000)
    c.add_argument(
        "--remote", type=int, nargs="+", default=[0], help="1 wedges the card (README)"
    )
    c = sub.add_parser("x280", help="an X280 thread decides every swap")
    c.add_argument("--mode", choices=sorted(MODES), default="self")
    c.add_argument("--swaps", type=int, default=1000, help="per Tensix core")
    c.add_argument("--cores", type=int, default=1, help="0: every core")
    c.add_argument("--clusters", type=int, nargs="+", default=[1])
    c.add_argument("--sizes", type=int, nargs="+", default=[0], help="KiB; 0: 3.5")
    c.add_argument("--deadline", type=float, default=30.0)
    args = p.parse_args()
    jobs = {"capture": capture, "swap": swap, "sizes": sizes, "fan": fan}
    jobs["x280"], jobs["stale"] = x280_job, stale
    jobs[args.job](args)


if __name__ == "__main__":
    main()
