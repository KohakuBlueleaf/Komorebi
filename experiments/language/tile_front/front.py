"""SiLU body experiment: the compute body komorebi.dsl.tile emits for `x * sigmoid(x)`
against the hand body (tt-metal's silu_tile) on one core; see README.md. Usage:
`python front.py build` (mock device: both bodies built, their words compared) or
`python front.py card [--tiles 16] [--seed 1]` (one core: bits, cycles of both)."""

import argparse
import difflib
import hashlib
import json
import os
import re
import secrets
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

import numpy as np

import komorebi as km
from komorebi.device import native
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.dsl.tile.unit import OPS, Insn, decode
from komorebi.program import several
from komorebi.session import kernels as session_kernels
from komorebi.session import process

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
GENERATED = KERNELS / "generated"
BODIES = {0: "hand", 1: "emitted"}
# front.hpp's layout.
CB_IN, CB_OUT, CB_INFO, MAX_TILES = 0, 16, 24, 32
NONCE, ERROR, TILES, BODY, LOOP, SYNC, SYNCS = 0, 1, 2, 64, 128, 192, 16
TILE_WORDS = 1024
OBJDUMP = "runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)


def log(message):
    print(message, flush=True)


def silu_unit():
    """The unit program of `y = silu(x)` over a stream of fp32 tiles."""
    x, y = TensorParam("x", False, F32), TensorParam("y", True, F32)
    ir = KernelIR("silu", [x, y], [], {})
    with trace.Tracer(ir):
        xs, ys = trace.TensorProxy(x), trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(xs)):
            trace.store(ys.tiles[t], trace.silu(trace.load(xs.tiles[t])))
    return ir, unit_program(ir)


def generate():
    """Writes the emitted header into kernels/generated; returns the unit program."""
    ir, unit = silu_unit()
    text = emit(unit)
    GENERATED.mkdir(exist_ok=True)
    (GENERATED / "silu.hpp").write_text(text)
    log(ir.dump())
    log(unit.dump())
    log(f"header sha1 {hashlib.sha1(text.encode()).hexdigest()[:12]}")
    return unit


def aiclk_mhz():
    """AICLK of the card in MHz from the driver's sysfs (TT_VISIBLE_DEVICES card)."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def launch(session, body, inputs):
    """One program on core (0, 0) running BODY `body` over `inputs` ([tiles, 1024]
    uint32 words of fp32); returns (the Launch, its output tensor, its info tensor,
    the nonce)."""
    tiles = len(inputs)
    nonce = secrets.randbits(32) | 1
    t_in = session.tensor(inputs, "uint32")
    t_out = session.tensor(np.zeros_like(inputs), "uint32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core = [(0, 0, 0, 0)]
    run = session.launch()
    for index in (CB_IN, CB_OUT):
        run.cb(core, index, "float32", 4 * TILE_WORDS, 4 * TILE_WORDS * tiles)
    run.cb(core, CB_INFO, "float32", 4 * TILE_WORDS, 4 * TILE_WORDS)

    def args(words):
        return [(0, 0, list(words))]

    kernel = str(KERNELS / "reader.cpp")
    run.kernel(
        kernel, core, 1, t_in.accessor_args(), [], [], args([t_in.address, tiles])
    )
    kernel = str(KERNELS / "writer.cpp")
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, tiles]
    run.kernel(kernel, core, 0, accessors, [], [], args(words))
    config = several.compute_config(
        {
            "math_fidelity": "HiFi4",
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {CB_IN: "UnpackToDestFp32"},
        }
    )
    run.kernel(
        str(KERNELS / "compute.cpp"),
        core,
        2,
        [],
        [("BODY", str(body))],
        [str(GENERATED)],
        args([nonce, tiles]),
        **config,
    )
    return run, t_out, t_info, nonce


def inputs_of(tiles, seed):
    """fp32 words: normal values scaled by 8, one tile over exponents -40 .. 40 with
    random signs, one tile of edges (0, -0, the overflow and underflow of e^-x)."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((tiles, TILE_WORDS)) * 8).astype(np.float32)
    if tiles > 1:
        e = rng.uniform(-40, 40, TILE_WORDS)
        x[1] = (np.sign(rng.standard_normal(TILE_WORDS)) * 2.0**e).astype(np.float32)
    if tiles > 2:
        edges = [0.0, -0.0, 1.0, -1.0, 87.0, -87.0, 88.8, -88.8, 89.0, -89.0, 104.0]
        edges += [-104.0, 1e-30, -1e-30, 1e30, -1e30, 3e38, -3e38]
        x[2, : len(edges)] = edges
        x[2, len(edges) :] = np.linspace(-100, 100, TILE_WORDS - len(edges))
    return x.view(np.uint32)


# ---- instruction words of a built math CPU -------------------------------------------


def words(elf, tree):
    """The coprocessor instruction words of an ELF in address order: (address, word)
    with the .ttinsn rotation undone (an instruction whose low two bits are not 3)."""
    text = subprocess.run(
        [str(Path(tree) / OBJDUMP), "-d", str(elf)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    out = []
    for line in text.splitlines():
        m = re.match(r"\s+([0-9a-f]+):\s+([0-9a-f]{8})\s", line)
        if not m:
            continue
        w = int(m.group(2), 16)
        if w & 3 != 3:
            out.append((int(m.group(1), 16), (w >> 2) | (w & 3) << 30))
    return out


def keys(stream):
    """Per instruction: the exact key (name, args) and the key modulo registers
    (name, the other args, how far back each register read was written)."""
    exact, renamed, last = [], [], {}
    for i, (_, word) in enumerate(stream):
        got = decode(word)
        if got is None:
            exact.append(("?", word))
            renamed.append(("?", word >> 24))
            continue
        name, args = got
        insn = Insn(name, args)
        exact.append((name, tuple(args.items())))
        regs = set(OPS[name].reads + OPS[name].writes)
        other = tuple((k, v) for k, v in args.items() if k not in regs)
        reads = [  # implied LRegs come by both names ("L0" and 0): the numbers
            r if r >= 8 else (i - last[r] if r in last else "in")
            for r in insn.reads()
            if isinstance(r, int)
        ]
        for r in insn.writes():
            last[r] = i
        renamed.append((name, other, tuple(reads)))
    return exact, renamed


def show(stream, i):
    got = decode(stream[i][1])
    where = "replay" if stream[i][0] is None else f"{stream[i][0]:05x}"
    return f"{where} {got[0]} {got[1]}" if got else hex(stream[i][1])


def expand(stream):
    """The stream as the coprocessor runs it: each REPLAY that plays is followed by
    the words it plays (address None), so register distances cross the replays."""
    out, buffer, recording = [], {}, []
    for address, word in stream:
        out.append((address, word))
        got = decode(word)
        if recording and not (got and got[0] == "REPLAY"):
            index, left = recording
            buffer[index] = word
            recording = [index + 1, left - 1] if left > 1 else []
            continue
        if got and got[0] == "REPLAY":
            a = got[1]
            if a["load_mode"]:
                recording = [a["start_idx"], a["len"]]
            else:
                played = range(a["start_idx"], a["start_idx"] + a["len"])
                out += [(None, buffer[i]) for i in played]
    return out


def start_of_tile(exact):
    """Index of the tile body's first face (its first REPLAY), else 0."""
    names = [k[0] for k in exact]
    return names.index("REPLAY") if "REPLAY" in names else 0


def compare(hand, emitted):
    """Exact and modulo-register comparison of two instruction streams."""
    stored = [len(hand), len(emitted)]
    hand, emitted = expand(hand), expand(emitted)
    he, hr = keys(hand)
    ee, er = keys(emitted)
    hb, eb = start_of_tile(he), start_of_tile(ee)
    out = {
        "words_stored": stored,
        "hand_words_run": len(hand),
        "emitted_words_run": len(emitted),
        "equal_exact": he == ee,
        "equal_modulo_registers": hr == er,
        "tile_words": [len(he) - hb, len(ee) - eb],
        "tile_equal_exact": he[hb:] == ee[eb:],
        "tile_equal_modulo_registers": keys(hand[hb:])[1] == keys(emitted[eb:])[1],
        "hand_ops": Counter(k[0] for k in he),
        "emitted_ops": Counter(k[0] for k in ee),
        "differences": [],
    }
    matcher = difflib.SequenceMatcher(a=hr, b=er, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out["differences"].append(
            {
                "kind": tag,
                "hand": [show(hand, i) for i in range(i1, i2)],
                "emitted": [show(emitted, j) for j in range(j1, j2)],
            }
        )
    return out


def built_elfs(cache):
    """{BODY: trisc1.elf} of the compute kernels built in JIT cache `cache`."""
    found = {}
    for defines in Path(cache).glob("**/kernels/compute/*/defines_generated.h"):
        m = re.search(r"#define BODY (\d)", defines.read_text())
        if m:
            found[int(m.group(1))] = defines.parent / "trisc1" / "trisc1.elf"
    return found


def build(args):
    """Mock device: both bodies built by the runtime's JIT, then compared."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    cache = HERE.parents[2] / ".logs" / "jit-front" / "mock"
    os.environ["TT_METAL_CACHE"] = str(cache)
    generate()
    inputs = inputs_of(args.tiles, args.seed)
    sizes = {}
    with km.Session() as session:
        for body in BODIES:
            log(f"build {BODIES[body]}")
            run, *_ = launch(session, body, inputs)
            _, *built = run.build()
            got = session_kernels.kernel_sizes(built)
            sizes[BODIES[body]] = [got.total, got.limit, got.text]
            log(f"built {BODIES[body]}: kernel buffer {got.total} of {got.limit} B")
    elfs = built_elfs(cache)
    streams = {b: words(elfs[b], tree) for b in BODIES}
    result = compare(streams[0], streams[1])
    for d in result["differences"]:
        log(f"-- {d['kind']}")
        log("   hand:    " + "\n            ".join(d["hand"] or ["(none)"]))
        log("   emitted: " + "\n            ".join(d["emitted"] or ["(none)"]))
    result["hand_ops"] = dict(result["hand_ops"])
    result["emitted_ops"] = dict(result["emitted_ops"])
    result["differences"] = len(result["differences"])
    result["elfs"] = {BODIES[b]: str(e) for b, e in elfs.items()}
    result["kernel_buffer"] = sizes
    print(json.dumps(result), flush=True)
    return 0


def check(inputs, outputs, infos, nonces):
    """The card run's record: bits of both bodies, error against float64, cycles."""
    x = inputs.view(np.float32).astype(np.float64)
    rec = {"tiles": len(inputs)}
    for body, info in infos.items():
        name = BODIES[body]
        rec[f"{name}_nonce_ok"] = int(info[NONCE]) == nonces[body]
        rec[f"{name}_error"] = int(info[ERROR])
        n = min(int(info[TILES]), MAX_TILES)
        if n == 0:
            continue
        sync = statistics.median(int(v) for v in info[SYNC : SYNC + SYNCS])
        spans = [int(v) for v in info[BODY : BODY + n]]
        loops = [int(v) for v in info[LOOP : LOOP + n]]
        rec[f"{name}_sync_span"] = sync
        rec[f"{name}_body_cycles"] = {
            "min": min(spans),
            "median": statistics.median(spans),
            "max": max(spans),
            "less_sync": statistics.median(spans) - sync,
        }
        rec[f"{name}_tile_cycles_median"] = statistics.median(loops[1:] or loops)
    hand, made = outputs[0], outputs[1]
    rec["bits_equal"] = bool(np.array_equal(hand, made))
    rec["words_differing"] = int((hand != made).sum())
    with np.errstate(over="ignore", invalid="ignore"):
        ref = x / (1 + np.exp(-x))
    y = made.view(np.float32).astype(np.float64)
    finite = np.isfinite(ref) & (np.abs(ref) > 1e-30)
    rel = np.abs(y[finite] - ref[finite]) / np.abs(ref[finite])
    rec["emitted_rel_error"] = {"max": float(rel.max()), "mean": float(rel.mean())}
    return rec


def card(args):
    """One core: both bodies over the same tiles; bits and cycles."""
    generate()
    inputs = inputs_of(args.tiles, args.seed)
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    started = time.time()
    outputs, infos, nonces, clock = {}, {}, {}, [aiclk_mhz()]
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for body in BODIES:
            run, t_out, t_info, nonce = launch(session, body, inputs)
            log(f"launch {BODIES[body]}")
            run.run()
            session.device.finish()
            log(f"done {BODIES[body]}; reading back")
            outputs[body] = t_out.read().reshape(len(inputs), TILE_WORDS).copy()
            infos[body] = t_info.read().reshape(-1).copy()
            nonces[body] = nonce
            log(f"read {BODIES[body]}")
            clock.append(aiclk_mhz())
    rec = {"aiclk_mhz": clock, "load": load, **check(inputs, outputs, infos, nonces)}
    rec["wall_s"] = round(time.time() - started, 1)
    print(json.dumps(rec), flush=True)
    clean = all(
        rec[f"{n}_nonce_ok"] and rec[f"{n}_error"] == 0 for n in BODIES.values()
    )
    return 0 if clean else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--tiles", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.tiles <= MAX_TILES:
        p.error(f"--tiles is 1 .. {MAX_TILES}")
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
