"""What an overlay costs inside the runtime: calls through a table or compiled in,
linking for a slot, parking per command, buffers set up again on a live core.
Progress lines, then one JSON line. Example: python costs.py capture; ... calls"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROBE = ROOT / "experiments" / "code" / "overlay_probe"
OUT = ROOT / "build" / "overlay_costs"
# One JIT cache for every job: the persistent ELFs the overlays link against are the
# ones that run. Set before tt-metal is loaded.
os.environ["TT_METAL_CACHE"] = str(OUT / "jit-cache")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROBE))

import builder  # noqa: E402
import numpy as np  # noqa: E402

from komorebi.device import native  # noqa: E402
from komorebi.session.process import aiclk_mhz  # noqa: E402
from komorebi.session.session import Session  # noqa: E402
from komorebi.session.sizes import KERNEL_BUFFER_BASE  # noqa: E402

T0 = time.monotonic()
CONTROL_WORDS = 128  # costs.h
EXPECT = 96  # costs.h: the expected output values
RESP_WORDS = 16
OP_CLOCK, OP_EXIT, OP_READ, OP_COMPUTE, OP_SWITCH, OP_LAYOUTS = 1, 2, 3, 4, 5, 6
FLAG_AT, FLAG_OVERLAY = 1, 2
STATUS = {
    0: "ok",
    1: "bad op",
    2: "read timeout",
    3: "too big",
    4: "idle",
    5: "cores not parked",
    6: "wait timeout",
}
SLOTS = 6  # data-movement 0, data-movement 1, compute 0, 1, 2, a second slot of 0
SLOT_BYTES = 64 * 1024
SECOND = 5
IDLE_UNITS = 2000  # the kernels leave after this many 2^20 cycles without a command
COMMANDS = OUT / "build-commands.txt"
TILE_BYTES = 2048
PAGES = 1024  # pages of the reader's DRAM tensor
CB_IN, CB_OUT, CB_READ = 0, 16, 2  # costs.h
TRISCS = ("trisc0", "trisc1", "trisc2")  # unpack, math, pack
PART_CORES = ("brisc", "ncrisc", *TRISCS)  # a tile's five cores, in slot order
KERNELS = {"brisc": "host", "ncrisc": "drain", **dict.fromkeys(TRISCS, "compute")}
# File-static library state a compute part keeps its own copy of and resets at entry
RESETS = ("_ZN7ckernel4mathL16src_zero_flag_hwE",)
LEAD_S = 0.02  # a broadcast starts this long after the clock was read
SCRIPT = builder.SCRIPT.replace(
    "  .data : {{",
    "  .got : {{ *(.got .got.*) }} > SLOT\n  .data : {{",
)
PIC = ["-fpie"]  # the position-independent form's compile flags
LAYOUT_A = ((CB_IN, 0, 2), (CB_OUT, 2, 2))  # (buffer, first page, pages)
LAYOUT_B = ((CB_IN, 4, 6), (CB_OUT, 10, 4))
LAYOUT_PAGES = 14


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


def build(commands, name, source, header, slot, elf, extra=(), resets=()):
    """Compiles `source` (with `header` as overlay_gen.h and `extra` flags), links it
    for [slot, slot + SLOT_BYTES) against the persistent ELF, checks it (the overlay
    probe's builder) and returns the Overlay."""
    key = hashlib.sha256(
        f"{commands}{header}{slot}{SLOT_BYTES}{extra}{SCRIPT}".encode()
        + Path(elf).read_bytes()
        + b"".join(p.read_bytes() for p in sorted(HERE.iterdir()) if p.is_file())
    ).hexdigest()[:12]
    work = OUT / "overlays" / f"{name}-{key}"
    work.mkdir(parents=True, exist_ok=True)
    (work / "overlay_gen.h").write_text(header)
    obj, out, script = work / "body.o", work / "overlay.elf", work / "overlay.ld"
    script.write_text(SCRIPT.format(slot=slot, size=SLOT_BYTES))
    if not out.exists():
        log(f"build {name} at {slot:#x}")
        build_directory = str(Path(elf).parent)
        flags = [
            a.replace(builder.BUILD_DIR, build_directory) for a in commands.compile
        ]
        builder.run(
            [
                *flags,
                *extra,
                f"-I{work}",
                f"-I{HERE}",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            cwd=work,
        )
        undefined = builder.run([builder.tool(commands, "gcc-nm"), "-u", str(obj)])
        builder.run(
            [
                *commands.link,
                f"-T{script}",
                f"-Wl,--just-symbols={elf}",
                f"-Wl,-Map={work / 'overlay.map'}",
                "-Wl,--cref",
                str(obj),
                "-o",
                str(out),
            ],
            cwd=work,
        )
        names = [u for u in undefined.split() if u != "U"]
        (work / "undefined.txt").write_text("\n".join(names))
    return builder.check(commands, name, out, slot, SLOT_BYTES, elf, work, resets)


def chain(steps, seed):
    st = builder.chain_steps(steps, seed)
    return builder.chain_header(st), (lambda arg: builder.chain_result(st, arg))


def steps_for(target):
    """Chain steps for an image of about `target` bytes (28 bytes a step, measured)."""
    return max(4, (target - 120) // 28)


def bf16_bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0]) >> 16


def f32_bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0])


def expect_words(scalar):
    """The 16 output values of a tile (inputs 1 .. 16 times the scalar) as 8 words."""
    values = [bf16_bits(v * scalar) for v in range(1, 17)]
    return [values[2 * i] | values[2 * i + 1] << 16 for i in range(8)]


def mhz_now(card):
    return aiclk_mhz(card)


def finish(job, out, suffix=""):
    out["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    text = json.dumps(out, default=lambda v: v.item())  # numpy scalars
    (OUT / f"{job}{suffix}.json").write_text(text)
    print(text, flush=True)


def stats(values, mhz=None):
    a = np.asarray(values, np.float64)
    out = {
        "n": len(a),
        "median": round(float(np.median(a)), 1),
        "min": round(float(a.min()), 1),
        "max": round(float(a.max()), 1),
    }
    if mhz:
        out["median_microseconds"] = round(float(np.median(a)) / mhz, 3)
    return out


class Clock:
    """AICLK samples (MHz) of the card, taken between commands."""

    def __init__(self, card):
        self.card, self.samples = card, []

    def sample(self):
        self.samples.append(mhz_now(self.card))
        return self.samples[-1]

    def summary(self):
        a = np.asarray(self.samples or [0], np.float64)
        return {"min": float(a.min()), "mean": round(float(a.mean()), 1)}


class Rig:
    """An open card with the five persistent kernels on `cores`: L1 control words,
    six slots, two buffer configurations and their pages; a DRAM response page per
    core and the reader's pages."""

    def __init__(self, session, cores):
        self.s, self.dev, self.cores = session, session.device, cores
        self.module = native.load_module()
        self.control = l1_rows(session, cores, CONTROL_WORDS)
        self.slots = l1_rows(session, cores, SLOTS * SLOT_BYTES // 4)
        self.layouts = l1_rows(session, cores, 2 * 128)
        self.pages = l1_rows(session, cores, LAYOUT_PAGES * TILE_BYTES // 4)
        self.results = dram_rows(session, np.zeros((len(cores), RESP_WORDS)))
        rows = np.arange(PAGES * TILE_BYTES // 4, dtype=np.uint32).reshape(PAGES, -1)
        rows[:, 0] = (np.arange(PAGES, dtype=np.uint32) * 2654435761) & 0xFFFFFFFF
        self.blocks = dram_rows(session, rows)
        self.first_words = rows[:, 0].astype(np.int64)
        if self.blocks.aligned_page_bytes != TILE_BYTES:
            raise ValueError("a reader page is read whole into one buffer page")
        self.launch = self.launcher()
        built = self.launch.build()
        elves = {Path(src).name: [Path(e) for e in es if e] for src, es in built[0]}
        self.elves = {
            "brisc": elves["host.cpp"][0],
            "ncrisc": elves["drain.cpp"][0],
            **dict(zip(TRISCS, elves["compute.cpp"], strict=True)),
        }
        self.commands = None  # read at the first build: capture writes them
        kernel, _, _ = self.module.kernel_buffer(self.dev)
        l1_base = int(kernel) + KERNEL_BUFFER_BASE
        reserved = [("firmware, mailboxes and the kernel buffer", 0, l1_base)]
        for name, buf, words in (
            ("the control words", self.control, CONTROL_WORDS),
            ("the configurations", self.layouts, 256),
            ("the buffer pages", self.pages, LAYOUT_PAGES * TILE_BYTES // 4),
        ):
            reserved.append((name, buf.address, buf.address + 4 * words))
        for k in range(SLOTS):
            builder.check_slot(self.slot(k), SLOT_BYTES, reserved)
        self.sequence = [0] * len(cores)
        log(
            f"slots {self.slot(0):#x} + {SLOTS} x {SLOT_BYTES}, control "
            f"{self.control.address:#x}, L1 buffers from {l1_base:#x}"
        )

    def slot(self, k):
        return self.slots.address + k * SLOT_BYTES

    def launcher(self):
        launch = self.s.launch()
        where = core_set(self.cores)
        args, rest = [], []
        for i, (x, y) in enumerate(self.cores):
            bank, at = page_place(self.results, i)
            words = [self.control.address, bank, at, self.slots.address, SLOT_BYTES]
            words += [IDLE_UNITS, self.blocks.address, self.blocks.aligned_page_bytes]
            args.append((x, y, words))
            others = [self.control.address, self.slots.address, SLOT_BYTES, IDLE_UNITS]
            rest.append((x, y, [*others, IDLE_UNITS]))
        includes = [str(HERE)]
        launch.kernel(str(HERE / "host.cpp"), where, 0, [], [], includes, args)
        launch.kernel(str(HERE / "drain.cpp"), where, 1, [], [], includes, rest)
        launch.kernel(str(HERE / "compute.cpp"), where, 2, [], [], includes, rest)
        for cb, pages in ((CB_IN, 2), (CB_OUT, 2), (CB_READ, 4)):
            launch.cb(where, cb, "bfloat16", TILE_BYTES, pages * TILE_BYTES)
        return launch

    def build(self, core, name, source, header, slot, extra=()):
        """An overlay for `core` (brisc, ... trisc2) linked at slot `slot`."""
        if self.commands is None:
            text = COMMANDS.read_text()
            self.commands = {
                t: builder.parse_commands(text, KERNELS[t], t) for t in PART_CORES
            }
        resets = RESETS if core in TRISCS else ()
        return build(
            self.commands[core],
            name,
            source,
            header,
            self.slot(slot),
            self.elves[core],
            extra,
            resets,
        )

    def upload(self, images, copies=1):
        """Puts each image in DRAM `copies` times (copy j in bank j % 8): returns per
        image the (bank, address, bytes) of each copy."""
        width = max(len(i) for i in images)
        rows = np.zeros((len(images) * copies, width // 4), np.uint32)
        for n, image in enumerate(images):
            for j in range(copies):
                rows[n * copies + j, : len(image) // 4] = np.frombuffer(
                    image, np.uint32
                )
        log(f"upload {len(images)} images x {copies} of up to {width} B")
        tensor = dram_rows(self.s, rows)
        self.kept = getattr(self, "kept", []) + [tensor]
        return [
            [
                (*page_place(tensor, n * copies + j), len(images[n]))
                for j in range(copies)
            ]
            for n in range(len(images))
        ]

    def write_layouts(self):
        """The two configurations (4 words a buffer: address, bytes, pages, page
        bytes) into every core's configuration words; returns (address, mask) of each.
        """
        made = []
        words = np.zeros(256, np.uint32)
        for n, layout in enumerate((LAYOUT_A, LAYOUT_B)):
            mask = 0
            for cb, first, pages in layout:
                at = self.pages.address + first * TILE_BYTES
                words[128 * n + 4 * cb : 128 * n + 4 * cb + 4] = [
                    at,
                    pages * TILE_BYTES,
                    pages,
                    TILE_BYTES,
                ]
                mask |= 1 << cb
            made.append((self.layouts.address + 512 * n, mask))
        for x, y in self.cores:
            self.module.write_l1(
                self.dev, x, y, self.layouts.address, [int(w) for w in words]
            )
        return made

    def run(self):
        log("launch")
        self.launch.run()
        log("launched")

    def words(self, op, flags=0, arg=0, count=0, at=0, program=0, extra=None):
        """Command words 4..31 (costs.h): `extra` maps a word index to its value."""
        w = [0] * 28
        w[0:6] = [op, flags, arg & 0xFFFFFFFF, count, at, program]
        for index, value in (extra or {}).items():
            w[index - 4] = int(value) & 0xFFFFFFFF
        return w

    def send(self, i, words, expect=None):
        """Command words 4..31 (and the expected values, words 96..103), then the
        sequence word."""
        x, y = self.cores[i]
        self.sequence[i] += 1
        at = self.control.address
        if expect is not None:
            self.module.write_l1(self.dev, x, y, at + 4 * EXPECT, list(expect))
        self.module.write_l1(self.dev, x, y, at + 16, [int(w) for w in words])
        self.module.write_l1(self.dev, x, y, at, [self.sequence[i]])
        return self.sequence[i]

    def response(self, i, sequence, timeout=5.0):
        """Core i's response line to `sequence` (16 words), polled from DRAM."""
        by = time.monotonic() + timeout
        while True:
            (page,) = self.s.read(self.results, first=i, pages=1)
            words = page.reshape(-1).view(np.uint32)[:RESP_WORDS].astype(np.int64)
            if words[0] == sequence:
                if words[1]:
                    raise RuntimeError(
                        f"core {i}: {STATUS.get(int(words[1]))}: {words}"
                    )
                return words
            if time.monotonic() > by:
                raise TimeoutError(f"core {i}: no response to {sequence}: {words}")

    def call(self, i, *args, timeout=5.0, expect=None, **kwargs):
        words = self.words(*args, **kwargs)
        return self.response(i, self.send(i, words, expect), timeout)

    def broadcast(self, words_of, mhz, expect=None, timeout=10.0):
        """The command words_of(i) on every core i, all starting at one wall-clock
        count LEAD_S after core 0's clock was read; the response lines in core order."""
        now = int(self.call(0, OP_CLOCK)[2])
        start = (now + int(LEAD_S * mhz * 1e6)) % (1 << 32)
        at = self.control.address
        for i, (x, y) in enumerate(self.cores):
            words = words_of(i)
            words[1] |= FLAG_AT
            words[4] = start
            if expect is not None:
                self.module.write_l1(self.dev, x, y, at + 4 * EXPECT, list(expect))
            self.module.write_l1(self.dev, x, y, at + 16, [int(w) for w in words])
        for i, (x, y) in enumerate(self.cores):
            self.sequence[i] += 1
            self.module.write_l1(self.dev, x, y, at, [self.sequence[i]])
        by = time.monotonic() + timeout
        while True:
            (pages,) = self.s.read(self.results)
            words = pages.reshape(len(self.cores), -1).view(np.uint32)[:, :RESP_WORDS]
            if all(
                int(words[i, 0]) == self.sequence[i] for i in range(len(self.cores))
            ):
                bad = [i for i in range(len(self.cores)) if words[i, 1]]
                if bad:
                    raise RuntimeError(f"cores {bad[:8]}: status {words[bad[0]]}")
                return words.astype(np.int64)
            if time.monotonic() > by:
                raise TimeoutError("not every core responded")

    def exit(self):
        for i in range(len(self.cores)):
            self.send(i, self.words(OP_EXIT))
        for i in range(len(self.cores)):
            self.response(i, self.sequence[i])
        gx, gy = self.s.grid
        done = self.module.wait_grid(self.dev, gx, gy, 10.0)
        log(f"kernels left: {done}")


def cores_of(session, which):
    """One core, the first row or the whole grid (each a grid from (0, 0))."""
    gx, gy = session.grid
    rows = {"one": 0, "row": 1, "grid": gy}[which]
    return [(x, y) for y in range(rows) for x in range(gx)] or [(0, 0)]


def shared_state(overlay):
    """The persistent image's symbols the linked overlay refers to (the linker's
    cross reference table, after link-time optimisation): what a table must carry."""
    text = (overlay.elf.parent / "overlay.map").read_text()
    table = text.split("Cross Reference Table", 1)[1].splitlines()[2:]
    used, name, defined = set(), None, None
    for line in table:
        if not line.strip():
            continue
        if not line.startswith(" "):
            name, where = line.split(None, 1)
            defined = where.strip().endswith(".elf")
        elif defined and ".ltrans" in line:
            used.add(name)
    return sorted(used)


def capture(args):
    """Builds the kernels once in a fresh JIT cache with the commands logged (child
    process) and keeps the compile and link commands for the builder."""
    if args.child:
        with Session(card=args.card, log=log) as s:
            Rig(s, [(0, 0)])
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


def reader_overlays(rig, slot, extra=(), tag=""):
    """The reader overlay in its two forms for `slot`: table, inline."""
    return {
        form: rig.build(
            "brisc",
            f"reader-{form}{tag}-{slot}",
            HERE / "reader.cpp",
            f"#define OV_TABLE {int(form == 'table')}\n",
            slot,
            extra,
        )
        for form in ("table", "inline")
    }


def compute_overlays(rig):
    """The compute overlay in its two forms: per form, its three parts."""
    return {
        form: [
            rig.build(
                t,
                f"compute-{form}-{t}",
                HERE / "compute_body.cpp",
                f"#define OV_TABLE {int(form == 'table')}\n",
                2 + k,
            )
            for k, t in enumerate(TRISCS)
        ]
        for form in ("table", "inline")
    }


def read_words(rig, form, place, blocks, first, slot=0, program=0):
    """The command words of one reader run; `place` the overlay's (bank, address,
    bytes), unused for the whole kernel."""
    flags = 0 if form == "whole" else FLAG_OVERLAY
    extra = {10: slot, 11: first}
    if place:
        extra.update({16: place[0], 17: place[1], 18: place[2]})
    return rig.words(OP_READ, flags, 0, blocks, program=program, extra=extra)


def read_sum(rig, first, blocks):
    pages = (first + np.arange(blocks)) % PAGES
    return int(rig.first_words[pages].sum()) & 0xFFFFFFFF


def compute_words(rig, form, parts, tiles, scalar, program=0):
    """The command words of one compute run; `parts` the three compute parts'
    (bank, address, bytes), unused for the whole kernel."""
    flags = 0 if form == "whole" else FLAG_OVERLAY
    extra = {}
    for k, place in enumerate(parts or ()):
        for j in range(3):
            extra[16 + 3 * (2 + k) + j] = place[j]
    return rig.words(OP_COMPUTE, flags, f32_bits(scalar), tiles, 0, program, extra)


def calls(args):
    """Question 1: a reader loop and a compute loop, each as a whole kernel, as an
    overlay calling the runtime through a table, as an overlay with the calls inline."""
    with Session(card=args.card, log=log) as s:
        cores = cores_of(s, args.cores)
        rig = Rig(s, cores)
        readers = reader_overlays(rig, 0)
        computes = compute_overlays(rig)
        sizes = {
            "reader": {f: len(o.image) for f, o in readers.items()},
            "reader_text": {f: o.text for f, o in readers.items()},
            "compute": {f: [len(p.image) for p in ps] for f, ps in computes.items()},
            "compute_text": {f: [p.text for p in ps] for f, ps in computes.items()},
        }
        log(f"sizes: {sizes}")
        shared = {f"reader-{f}": shared_state(o) for f, o in readers.items()}
        for f, ps in computes.items():
            shared[f"compute-{f}"] = sorted({n for p in ps for n in shared_state(p)})
        log(f"shared state: {shared}")
        images = [readers["table"].image, readers["inline"].image]
        images += [p.image for f in ("table", "inline") for p in computes[f]]
        placed = rig.upload(images, copies=8 if len(cores) > 1 else 1)
        reader_place = {"table": placed[0], "inline": placed[1]}
        compute_place = {"table": placed[2:5], "inline": placed[5:8]}
        clock = Clock(args.card)
        mhz = clock.sample()
        rig.run()
        rows = {"reader": {}, "compute": {}}
        forms = ("whole", "table", "inline")
        ids = {"table": 1, "inline": 2}
        try:
            for rep in range(args.reps):
                for form in forms:

                    def reader_of(i, form=form, rep=rep):
                        place = None
                        if form != "whole":
                            place = reader_place[form][i % len(reader_place[form])]
                        first = (8 * i + 64 * rep) % PAGES
                        return read_words(
                            rig, form, place, args.blocks, first, 0, ids.get(form, 0)
                        )

                    resp = rig.broadcast(reader_of, mhz, timeout=20.0)
                    clock.sample()
                    wrong = 0
                    for i, r in enumerate(resp):
                        first = (8 * i + 64 * rep) % PAGES
                        wrong += int(r[2]) != read_sum(rig, first, args.blocks)
                    per = resp[:, 6] / args.blocks
                    rows["reader"].setdefault(form, []).append(
                        {"wrong": wrong, "per_block": per.tolist()}
                    )
                    log(
                        f"reader {form} rep {rep}: {np.median(per):.1f} cycles a block"
                        f" (max {per.max():.1f}), wrong {wrong}"
                    )
                for form in forms:
                    scalar = float(2 + rep % 5)

                    def compute_of(i, form=form, scalar=scalar):
                        parts = None
                        if form != "whole":
                            parts = [p[i % len(p)] for p in compute_place[form]]
                        return compute_words(
                            rig, form, parts, args.tiles, scalar, ids.get(form, 0)
                        )

                    resp = rig.broadcast(
                        compute_of, mhz, expect_words(scalar), timeout=20.0
                    )
                    clock.sample()
                    wrong = int(resp[:, 2].sum())
                    per = resp[:, 6] / args.tiles
                    rows["compute"].setdefault(form, []).append(
                        {"wrong": wrong, "per_tile": per.tolist()}
                    )
                    log(
                        f"compute {form} rep {rep}: {np.median(per):.1f} cycles a tile"
                        f" (max {per.max():.1f}), wrong tiles {wrong}"
                    )
            # the check's own control: expected values of another scalar
            resp = rig.broadcast(
                lambda i: compute_words(rig, "whole", None, args.tiles, 2.0),
                mhz,
                expect_words(3.0),
            )
            control = int(resp[:, 2].sum())
            log(f"control: {control} wrong tiles of {args.tiles * len(cores)}")
        finally:
            rig.exit()
    summary = {"control_wrong": control, "control_tiles": args.tiles * len(cores)}
    for loop, by_form in rows.items():
        key = "per_block" if loop == "reader" else "per_tile"
        summary[loop] = {
            form: {
                "wrong": sum(r["wrong"] for r in reps),
                "cycles": stats([v for r in reps for v in r[key]]),
                "worst_core_median": round(
                    float(np.median([max(r[key]) for r in reps])), 1
                ),
            }
            for form, by in by_form.items()
            for reps in [by]
        }
    finish(
        "calls",
        {
            "cores": len(cores),
            "blocks": args.blocks,
            "tiles": args.tiles,
            "reps": args.reps,
            "aiclk_mhz": clock.summary(),
            "sizes": sizes,
            "shared_state": shared,
            "summary": summary,
        },
        f"-{args.cores}",
    )


def slots(args):
    """Question 2: one reader body linked at a fixed slot, once per slot, and
    position-independent (one image for any slot); then two overlays held at once."""
    with Session(card=args.card, log=log) as s:
        cores = cores_of(s, args.cores)
        rig = Rig(s, cores)
        fixed = {k: reader_overlays(rig, k)["inline"] for k in (0, SECOND)}
        pic = {k: reader_overlays(rig, k, PIC, "-pic") for k in (0, SECOND)}
        same = {f: pic[0][f].image == pic[SECOND][f].image for f in pic[0]}
        if not all(same.values()):
            raise builder.OverlayError(f"not position-independent: {same}")
        sizes = {
            "fixed": [len(fixed[k].image) for k in (0, SECOND)],
            "pic": len(pic[0]["inline"].image),
            "pic_table": len(pic[0]["table"].image),
            "pic_text": pic[0]["inline"].text,
            "fixed_text": fixed[0].text,
        }
        log(f"sizes: {sizes}; position-independent images equal: {same}")
        copies = 8 if len(cores) > 1 else 1
        images = [fixed[0].image, fixed[SECOND].image]
        images += [pic[0]["inline"].image, pic[0]["table"].image]
        placed = rig.upload(images, copies)
        runs = [  # (form, slot, image index)
            ("fixed", 0, 0),
            ("fixed", SECOND, 1),
            ("pic", 0, 2),
            ("pic", SECOND, 2),
        ]
        clock = Clock(args.card)
        mhz = clock.sample()
        rig.run()
        rows, program = {}, 100
        try:
            for rep in range(args.reps):
                for form, slot, image in runs:
                    for held in (False, True):
                        program += 0 if held else 1
                        first = 64 * rep % PAGES

                        def words_of(i, slot=slot, image=image, p=program, f=first):
                            place = placed[image][i % copies]
                            return read_words(
                                rig, "overlay", place, args.blocks, f, slot, p
                            )

                        resp = rig.broadcast(words_of, mhz, timeout=20.0)
                        want = read_sum(rig, first, args.blocks)
                        row = rows.setdefault(f"{form}-{slot}", {"wrong": 0})
                        row["wrong"] += int(sum(int(r[2]) != want for r in resp))
                        row["wrong"] += int(sum(int(r[4]) != (not held) for r in resp))
                        key = "held_per_block" if held else "load"
                        values = resp[:, 6] / args.blocks if held else resp[:, 5]
                        row.setdefault(key, []).extend(values.tolist())
                clock.sample()
                log(
                    f"slots rep {rep}: wrong "
                    + ", ".join(f"{k} {v['wrong']}" for k, v in rows.items())
                )
            # two overlays held at once: the inline one in slot 0, the table one in
            # the second slot, alternated with no load after the first two
            loads = wrong = 0
            for n in range(args.alternations):
                slot, image = (0, 2) if n % 2 == 0 else (SECOND, 3)
                first = (7 * n) % PAGES

                def words_of(i, slot=slot, image=image, first=first):
                    place = placed[image][i % copies]
                    return read_words(
                        rig, "overlay", place, 64, first, slot, 900 + image
                    )

                resp = rig.broadcast(words_of, mhz, timeout=20.0)
                loads += int(resp[:, 4].sum())
                wrong += int(sum(int(r[2]) != read_sum(rig, first, 64) for r in resp))
                if n % 200 == 0:
                    log(f"two slots {n}: loads {loads}, wrong {wrong}")
            clock.sample()
        finally:
            rig.exit()
    summary = {
        k: {
            "wrong": v["wrong"],
            "load_cycles": stats(v["load"], mhz),
            "cycles_per_block": stats(v["held_per_block"]),
        }
        for k, v in rows.items()
    }
    finish(
        "slots",
        {
            "cores": len(cores),
            "blocks": args.blocks,
            "aiclk_mhz": clock.summary(),
            "sizes": sizes,
            "position_independent_equal": same,
            "runs": summary,
            "two_slots": {
                "alternations": args.alternations,
                "loads": loads,
                "loads_expected": 2 * len(cores),
                "wrong": wrong,
            },
        },
        f"-{args.cores}",
    )


def park(args):
    """Question 3: a command naming the program the core holds (the check only), and
    one naming another (park, five loads, invalidations, release), by part size."""
    with Session(card=args.card, log=log) as s:
        cores = cores_of(s, args.cores)
        rig = Rig(s, cores)
        copies = 8 if len(cores) > 1 else 1
        programs, images = [], []
        for kib in args.sizes:
            for which in range(2):
                parts, refs = [], []
                for k, t in enumerate(PART_CORES):
                    header, ref = chain(
                        steps_for(kib * 1024), 1000 * kib + 10 * which + k
                    )
                    parts.append(
                        rig.build(
                            t, f"part{kib}-{which}-{t}", HERE / "part.cpp", header, k
                        )
                    )
                    refs.append(ref)
                programs.append(
                    (kib, which, len(images), refs, [len(p.image) for p in parts])
                )
                images += [p.image for p in parts]
                log(f"program {kib} KiB {which}: parts {programs[-1][4]} B")
        placed = rig.upload(images, copies)
        clock = Clock(args.card)
        mhz = clock.sample()
        rig.run()
        rows = {}
        rng = np.random.default_rng(3)
        try:
            for kib in args.sizes:
                pair = [p for p in programs if p[0] == kib]
                for mode in ("switch", "check"):
                    row = rows.setdefault(f"{kib}-{mode}", {"wrong": 0, "loads": 0})
                    for n in range(args.reps):
                        _, which, first, refs, _ = pair[
                            n % 2 if mode == "switch" else 0
                        ]
                        program = 10 * kib + which + 1
                        arg = int(rng.integers(0, 1 << 32))

                        def words_of(i, first=first, program=program, arg=arg):
                            extra = {}
                            for k in range(5):
                                place = placed[first + k][i % copies]
                                for j in range(3):
                                    extra[16 + 3 * k + j] = place[j]
                            return rig.words(OP_SWITCH, 0, arg, 0, 0, program, extra)

                        resp = rig.broadcast(words_of, mhz, timeout=20.0)
                        for r in resp:
                            got = [int(r[2]), *(int(v) for v in r[8:12])]
                            row["wrong"] += sum(g != f(arg) for g, f in zip(got, refs))
                        row["loads"] += int(resp[:, 4].sum())
                        if n == 0 and mode == "check":
                            continue  # the first may load: not a check
                        row.setdefault("to_go", []).extend(resp[:, 6].tolist())
                        row.setdefault("release", []).extend(resp[:, 12].tolist())
                        row.setdefault("to_go_last_core", []).append(
                            int(resp[:, 6].max())
                        )
                        if mode == "switch":
                            row.setdefault("loaded", []).extend(resp[:, 5].tolist())
                    clock.sample()
                    log(
                        f"park {kib} KiB {mode}: wrong {row['wrong']}, loads "
                        f"{row['loads']}, to go {np.median(row['to_go']):.0f} cycles"
                    )
        finally:
            rig.exit()
    summary = {
        k: {
            "wrong": v["wrong"],
            "loads": v["loads"],
            "to_go": stats(v["to_go"], mhz),
            "release": stats(v["release"], mhz),
            "to_go_last_core": stats(v["to_go_last_core"], mhz),
            **({"loaded": stats(v["loaded"], mhz)} if "loaded" in v else {}),
        }
        for k, v in rows.items()
    }
    finish(
        "park",
        {
            "cores": len(cores),
            "dram_copies": copies,
            "aiclk_mhz": clock.summary(),
            "part_bytes": {f"{p[0]}-{p[1]}": p[4] for p in programs},
            "runs": summary,
        },
        f"-{args.cores}",
    )


def layouts(args):
    """Question 4: two buffer configurations switched on a live core between runs of
    tiles through compute, with every output value checked."""
    with Session(card=args.card, log=log) as s:
        cores = cores_of(s, args.cores)
        rig = Rig(s, cores)
        (a, mask_a), (b, mask_b) = rig.write_layouts()
        log(f"configurations at {a:#x} ({mask_a:#x}), {b:#x} ({mask_b:#x})")
        clock = Clock(args.card)
        mhz = clock.sample()
        rig.run()
        done = {"switches": 0, "tiles": 0, "wrong": 0, "bad": 0, "cycles": 0}
        low, high = [], []
        try:
            for n in range(args.commands):
                scalar = float(2 + n % 5)
                extra = {11: a, 12: mask_a, 13: b, 14: mask_b, 15: args.tiles}

                def words_of(i, scalar=scalar, extra=extra):
                    return rig.words(
                        OP_LAYOUTS, 0, f32_bits(scalar), args.switches, 0, 0, extra
                    )

                resp = rig.broadcast(words_of, mhz, expect_words(scalar), timeout=60.0)
                clock.sample()
                for r in resp:
                    done["switches"] += args.switches
                    done["tiles"] += int(r[8])
                    done["wrong"] += int(r[2])
                    done["bad"] += int(r[11])
                    done["cycles"] += int(r[9]) | int(r[10]) << 32
                    low.append(int(r[5]))
                    high.append(int(r[6]))
                log(
                    f"layouts {n}: {done}, a switch {int(resp[0][5])} to "
                    f"{int(resp[0][6])} cycles"
                )
        finally:
            rig.exit()
    finish(
        "layouts",
        {
            "cores": len(cores),
            "aiclk_mhz": clock.summary(),
            "tiles_per_run": args.tiles,
            "layouts": {"a": LAYOUT_A, "b": LAYOUT_B},
            **done,
            "cycles_per_switch": round(done["cycles"] / max(done["switches"], 1), 1),
            "switch_min": min(low),
            "switch_max": max(high),
        },
        f"-{args.cores}",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--card", type=int, default=1)
    sub = p.add_subparsers(dest="job", required=True)
    c = sub.add_parser("capture", help="the kernels' build commands")
    c.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("calls", help="question 1: runtime calls from an overlay")
    c.add_argument("--cores", choices=("one", "row", "grid"), default="one")
    c.add_argument("--blocks", type=int, default=2048)
    c.add_argument("--tiles", type=int, default=1024)
    c.add_argument("--reps", type=int, default=5)
    c = sub.add_parser("slots", help="question 2: how an overlay is linked")
    c.add_argument("--cores", choices=("one", "row", "grid"), default="one")
    c.add_argument("--blocks", type=int, default=256)
    c.add_argument("--reps", type=int, default=20)
    c.add_argument("--alternations", type=int, default=1000)
    c = sub.add_parser("park", help="question 3: parking per command")
    c.add_argument("--cores", choices=("one", "row", "grid"), default="one")
    c.add_argument("--sizes", type=int, nargs="+", default=[4, 20, 60])
    c.add_argument("--reps", type=int, default=100)
    c = sub.add_parser("layouts", help="question 4: buffers set up again")
    c.add_argument("--cores", choices=("one", "row", "grid"), default="one")
    c.add_argument("--switches", type=int, default=1000)
    c.add_argument("--commands", type=int, default=10)
    c.add_argument("--tiles", type=int, default=8)
    args = p.parse_args()
    jobs = {
        "capture": capture,
        "calls": calls,
        "slots": slots,
        "park": park,
        "layouts": layouts,
    }
    jobs[args.job](args)


if __name__ == "__main__":
    main()
