"""Tile-rules experiment: bodies komorebi.dsl.tile emits against the hand bodies they
stand in for, on one core (README.md). `rules.py build`: a mock device builds every
body, the host checks run on made-up results; `rules.py card --cases ..`: values
against float64 and cycles of each body."""

import argparse
import json
import math
import os
import secrets
import statistics
import struct
import sys
import time
from pathlib import Path

os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")

import numpy as np

import komorebi as km
from komorebi import native
from komorebi import session as km_session
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, TensorParam
from komorebi.dsl.tile.rows import row_program
from komorebi.dsl.tile.steps import SLOT, WORD, step_program

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
GENERATED = KERNELS / "generated"
# rules.hpp's layout.
CB_A, CB_B, CB_P, CB_OUT, CB_INFO, MAX_UNITS = 0, 1, 2, 16, 24, 32
CB_SQUARES, CB_SCALE, CB_Y = 3, 4, 5  # the hand norm's
NONCE, ERROR, UNITS, BODY, LOOP, SYNC, SYNCS = 0, 1, 2, 64, 128, 192, 16
TILE_WORDS = 1024
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
SCALE = 0.08838834764831845  # 1 / sqrt(128), rounded to fp32 by both bodies
EPS = 1e-6
WIDTH = 2  # tiles in a row of the norm
P_TILE, ROWS_TILE = 4, 2  # the norm's Dst tiles: per-column operand, per-row value
# Largest relative error against float64 of an emitted body (README.md, "The gate");
# hand bodies are reported against it too.
TOLERANCE = {
    **{"silu": 1e-6, "add": 2**-23, "scale": 2**-23, "gate": 1e-6, "norm": 1e-5},
    "exp": 1e-5,
}
# The hand bodies held to their own bound: attention's exp polynomial is 1.75e-3.
HAND_TOLERANCE = {"exp": 2e-3}
# The exp the program states (the hand body's: 2e-3 over s scale - m <= 0); the
# "stated" form is emitted at it, held to it, the accurate one beside it.
STEP_EXP = {"error": 2e-3, "over": (-math.inf, 0.0)}
STATED_TOLERANCE = {"exp": STEP_EXP["error"]}
# The softmax step (base 2, both bodies' exp2 polynomial): P, the rescale factor and
# the sum within 2e-3, the reference m within 1e-6, the word's sign exact.
STEP_TOLERANCE = {"p": 2e-3, "alpha": 2e-3, "l": 2e-3, "m": 1e-6, "m_abs": 1e-4}
# (the hand body keeps 639 - A m: its m to within an fp32 step at 639, 6.1e-5)
STEP_WIDTH, STEP_LAG = 3, 16.0  # key tiles a step; the reference's lag (ktt's tau)
STEP_TILE = 3  # the state's Dst tile (the hand body's ST)
ATTENTION = HERE.parents[1] / "projects" / "attention" / "kernels"
EXP_SCALE = 0.125  # 1 / sqrt(64), exact in fp32
EXP_A = EXP_SCALE * 1.4426950408889634  # the hand body's A = scale log2 e
EXP_TILE = {
    12: 3,
    13: 1,
    16: 1,
}  # the row state's Dst tile: the hand body's, the emitted

# BODY: (case, form); "macro": the load-macro rule, "stated": at STEP_EXP, "accurate":
# the default exp. Hand: tt-metal's library, projects/decode, projects/attention.
BODIES = {
    0: ("silu", "hand"),
    1: ("silu", "emitted"),
    2: ("add", "hand"),
    3: ("add", "emitted"),
    4: ("add", "macro"),
    5: ("scale", "hand"),
    6: ("scale", "emitted"),
    7: ("scale", "macro"),
    8: ("gate", "hand"),
    9: ("gate", "emitted"),
    10: ("norm", "hand"),
    11: ("norm", "emitted"),
    12: ("exp", "hand"),
    13: ("exp", "emitted"),
    14: ("softmax", "hand"),
    15: ("softmax", "emitted"),
    16: ("exp", "stated"),
    17: ("softmax", "accurate"),
}
FORMS = ("hand", "emitted", "macro", "stated", "accurate")
CASES = ("silu", "add", "scale", "gate", "norm", "exp", "softmax")
# Run on the card already, over the same inputs (the tile-front experiment): built
# and checked here, not run again.
ALREADY_RUN = {0, 12}
INPUTS = {
    **{"silu": 1, "add": 2, "scale": 1, "gate": 2, "norm": 1, "exp": 1},
    "softmax": 1,
}


def log(message):
    print(message, flush=True)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def stream(name, inputs, fn):
    """The IR of y = fn(a, ..) over a stream of fp32 tiles."""
    params = [TensorParam(n, False, F32) for n in "abc"[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(p) for p in params]
        ys = trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            loads = [trace.load(p.tiles[t]) for p in proxies]
            trace.store(ys.tiles[t], fn(*loads))
    return ir


def norm_ir():
    """y = x rsqrt(mean(x^2) + eps) g over rows of WIDTH tiles, g one weight row."""
    x, g = TensorParam("x", False, F32), TensorParam("g", False, F32)
    y = TensorParam("y", True, F32)
    ir = KernelIR("norm", [x, g, y], [], {})
    with trace.Tracer(ir):
        xs, gs, ys = (trace.TensorProxy(t) for t in (x, g, y))
        weight = trace.resident(gs.tiles[0, :])
        for t in trace.grid(trace.num_tile_rows(xs)):
            row = trace.load(xs.tiles[t, :])
            ms = trace.reduce_sum(row * row, exact=True) * (1.0 / (32 * WIDTH))
            r = trace.rsqrt(ms + EPS)
            trace.store(ys.tiles[t, :], row * r.col() * weight.row())
    return ir


def expstep_ir(name="expstep", **exp):
    """p = exp(s scale - m), m a per-row value: the row state tile (each row's scaled
    reference in all its lanes), broadcast along the row with .col(); `exp`: the
    error the program states (trace.exp's error and over)."""
    s, m = TensorParam("s", False, F32), TensorParam("m", False, F32)
    p = TensorParam("p", True, F32)
    ir = KernelIR(name, [s, m, p], [], {})
    with trace.Tracer(ir):
        ss, ms, ps = (trace.TensorProxy(t) for t in (s, m, p))
        for t in trace.grid(trace.num_tiles(ss)):
            x, state = trace.load(ss.tiles[t]), trace.load(ms.tiles[t])
            trace.store(ps.tiles[t], trace.exp(x * EXP_SCALE - state.col(), **exp))
    return ir


def softmax_ir(name="softstep", exp=STEP_EXP):
    """The online softmax step in base 2 over a row of STEP_WIDTH score tiles: the
    new reference (lagging by up to STEP_LAG), the old state's rescale factor, P and
    the new sum; the state tile in and out (m, l, alpha) and whether a row moved."""
    s, st = TensorParam("s", False, F32), TensorParam("st", False, F32)
    p, out = TensorParam("p", True, F32), TensorParam("out", True, F32)
    ir = KernelIR(name, [s, st, p, out], [], {})
    with trace.Tracer(ir):
        ss, sts, ps, outs = (trace.TensorProxy(t) for t in (s, st, p, out))
        for t in trace.grid(trace.num_tile_rows(ss)):
            x = trace.load(ss.tiles[t, :]) * EXP_A
            state = trace.load_state(sts.tiles[t])
            m_old, l_old = state[0], state[1]
            r = trace.reduce_max(x, exact=True)
            m_new = trace.lagging_max(m_old, r, STEP_LAG)
            alpha = trace.exp2(m_old - m_new, **exp)
            prob = trace.exp2(x - m_new.col(), **exp)
            l_new = alpha * l_old + trace.reduce_sum(prob, exact=True)
            trace.store(ps.tiles[t, :], prob)
            moved = trace.word_max(m_new - m_old)
            trace.store_state(outs.tiles[t], [m_new, l_new, alpha], word=moved)
    return ir


def units():
    """{header name: unit program} of every emitted body."""
    out = {}
    fns = {
        "silu": (1, trace.silu),
        "add": (2, lambda a, b: a + b),
        "scale": (1, lambda x: x * SCALE),
        "gate": (2, lambda a, b: trace.silu(a) * b),
    }
    for name, (inputs, fn) in fns.items():
        out[name] = unit_program(stream(name, inputs, fn))
    for name in ("add", "scale"):
        inputs, fn = fns[name]
        unit = unit_program(stream(f"{name}_macro", inputs, fn), load_macro=True)
        out[f"{name}_macro"] = unit
    out["norm"] = row_program(norm_ir(), WIDTH, ROWS_TILE, P_TILE)
    out["expstep"] = unit_program(expstep_ir())
    out["expstep_flash"] = unit_program(expstep_ir("expstep_flash", **STEP_EXP))
    out["softstep"] = step_program(softmax_ir(), STEP_WIDTH, STEP_TILE)
    accurate = softmax_ir("softstep_accurate", {})
    out["softstep_accurate"] = step_program(accurate, STEP_WIDTH, STEP_TILE)
    return out


def generate():
    """Writes every emitted header into kernels/generated; returns the units' costs
    and the load-macro rule's verdict on the bodies it refuses."""
    GENERATED.mkdir(exist_ok=True)
    costs = {}
    for name, unit in units().items():
        (GENERATED / f"{name}.hpp").write_text(emit(unit))
        log(unit.dump())
        costs[name] = unit.cost("math")[1]
    refused = {}
    for name, inputs, fn in (
        ("silu", 1, trace.silu),
        ("gate", 2, lambda a, b: trace.silu(a) * b),
    ):
        unit = unit_program(stream(name, inputs, fn), load_macro=True)
        refused[name] = next(p.done for p in unit.passes if p.name == "load_macro")
    log(f"estimates {json.dumps(costs)}")
    log(f"load-macro refusals {json.dumps(refused)}")
    return costs


def aiclk_mhz():
    """AICLK of the card in MHz from the driver's sysfs (TT_VISIBLE_DEVICES card)."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        with open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def p_pages(body):
    """Pages of P: the emitted norm's per-column operand tile; the hand norm's
    identity, mean and weight tiles (decode's constant tiles, and g in every row)."""
    return {10: 2 + WIDTH, 11: 1}.get(body, int(body >= 12))


def setting(body):
    """The compile-time setting of BODY `body`."""
    case, form = BODIES[body]
    unit_tiles = {"norm": WIDTH, "softmax": STEP_WIDTH}.get(case, 1)
    copy_b = INPUTS[case] == 2 and (case, form) != ("gate", "hand")
    return {
        "BODY": body,
        "UNIT_TILES": unit_tiles,
        "PACK_STATE": int(case == "softmax"),
        "STEP_THRESHOLD": f"{bits(639.0 + STEP_LAG)}u",
        "INPUTS": INPUTS[case],
        "COPY_B": int(copy_b),
        "HAS_P": p_pages(body),
        "P_TILE": STEP_TILE if case == "softmax" else EXP_TILE.get(body, P_TILE),
        "EXP_A": f"{float(np.float32(EXP_A))!r}f",
        "SCALE_BITS": f"{bits(SCALE)}u",
        "INV_N_BITS": f"{bits(1.0 / (32 * WIDTH))}u",
        "EPS_BITS": f"{bits(EPS)}u",
    }


def launch(session, body, data):
    """One program on core (0, 0) running BODY `body` over `data` (a, b, p: uint32
    words of fp32 tiles); returns (the Launch, the output, the info tensor, nonce)."""
    case, form = BODIES[body]
    defines = setting(body)
    unit_tiles = defines["UNIT_TILES"]
    count = len(data["a"]) // unit_tiles
    nonce = secrets.randbits(32) | 1
    keys = ("a", "b", "p_hand" if form == "hand" else "p")
    tensors = [session.tensor(data[k], "uint32") for k in keys]
    out_tiles = count * (unit_tiles + defines["PACK_STATE"])
    t_out = session.tensor(np.zeros((out_tiles, TILE_WORDS), np.uint32), "uint32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core = [(0, 0, 0, 0)]
    page = 4 * TILE_WORDS
    run = session.launch()
    run.cb(core, CB_A, "float32", page, 2 * unit_tiles * page)
    run.cb(core, CB_B, "float32", page, 2 * page)
    run.cb(core, CB_P, "float32", page, max(1, defines["HAS_P"]) * page)
    run.cb(core, CB_SQUARES, "float32", page, page)
    run.cb(core, CB_SCALE, "float32", page, page)
    run.cb(core, CB_Y, "float32", page, 2 * unit_tiles * page)
    run.cb(core, CB_OUT, "float32", page, 2 * (unit_tiles + 1) * page)
    run.cb(core, CB_INFO, "float32", page, page)

    def args(words):
        return [(0, 0, list(words))]

    accessors = [w for t in tensors for w in t.accessor_args()]
    words = [t.address for t in tensors]
    words += [count, unit_tiles, INPUTS[case], defines["HAS_P"], int(body == 10)]
    run.kernel(str(KERNELS / "reader.cpp"), core, 1, accessors, [], [], args(words))
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, out_tiles]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], args(words))
    # Unpacked to Dst: the buffers copy_tile reads. Not the hand norm's P (the matrix
    # unit's operands: such a buffer unpacks Float32 into its source registers).
    unpack = {CB_A: "UnpackToDestFp32"}
    if defines["COPY_B"]:
        unpack[CB_B] = "UnpackToDestFp32"
    if body == 11 or body >= 12:
        unpack[CB_P] = "UnpackToDestFp32"
    options = {
        "math_fidelity": "HiFi4",
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": unpack,
    }
    if case == "norm":
        options["dst_full_sync_en"] = True
    run.kernel(
        str(KERNELS / "compute.cpp"),
        core,
        2,
        [],
        [(k, str(v)) for k, v in defines.items()],
        [str(GENERATED), str(ATTENTION)],
        args([nonce, count]),
        **km_session.compute_config(options),
    )
    return run, t_out, t_info, nonce


def tiles_of(matrix):
    """[32, 32 W] fp32 rows as W tiles of 1024 words in face order."""
    w = matrix.shape[1] // 32
    out = np.empty((w, TILE_WORDS), matrix.dtype)
    for t in range(w):
        block = matrix[:, 32 * t : 32 * t + 32]
        faces = [block[16 * (f // 2) :, 16 * (f % 2) :][:16, :16] for f in range(4)]
        out[t] = np.concatenate([face.reshape(-1) for face in faces])
    return out


def matrix_of(tiles):
    """The inverse of tiles_of."""
    w = len(tiles)
    out = np.empty((32, 32 * w), tiles.dtype)
    for t in range(w):
        faces = tiles[t].reshape(4, 16, 16)
        for f in range(4):
            r, c = 16 * (f // 2), 32 * t + 16 * (f % 2)
            out[r : r + 16, c : c + 16] = faces[f]
    return out


def p_tile(g):
    """The per-column operand tile of weight row g (64 values): face c, rows 8 t ..
    8 t + 3 hold g's columns 32 t + 16 c .. + 15; the rows below (beta) are 0."""
    out = np.zeros((4, 16, 16), np.float32)
    for t in range(WIDTH):
        for c in range(2):
            out[c, 8 * t : 8 * t + 4, :] = g[32 * t + 16 * c : 32 * t + 16 * c + 16]
    return out.reshape(1, TILE_WORDS)


def state_tile(slots):
    """The emitted step's state tile (fp32 words in face order) holding per-row
    values `slots` ({k: 32 values}): slot k of row group (h, q) at Dst + 16 h + 4 q +
    SLOT[k], lane l for row 16 h + 4 q + l / 8 (steps.py)."""
    dst = np.zeros((64, 16), np.float32)
    lanes = np.arange(32)
    for k, values in slots.items():
        for h in range(2):
            for q in range(4):
                address = 16 * h + 4 * q + SLOT[k]
                rows = (address & ~3) + lanes // 8
                cols = 2 * (lanes % 8) + (1 if address & 2 else 0)
                dst[rows, cols] = np.asarray(values)[16 * h + 4 * q + lanes // 8]
    return dst.reshape(1, TILE_WORDS)


def state_slot(tile, k):
    """(32 values of slot k, the largest spread among the 8 lanes of a row)."""
    dst = tile.view(np.float32).reshape(64, 16).astype(np.float64)
    lanes, out, spread = np.arange(32), np.zeros(32), 0.0
    for h in range(2):
        for q in range(4):
            address = 16 * h + 4 * q + SLOT[k]
            rows = (address & ~3) + lanes // 8
            cols = 2 * (lanes % 8) + (1 if address & 2 else 0)
            got = dst[rows, cols].reshape(4, 8)
            out[16 * h + 4 * q : 16 * h + 4 * q + 4] = got[:, 0]
            spread = max(spread, float(np.abs(got - got[:, :1]).max()))
    return out, spread


def inputs_of(case, count, seed):
    """{a, b, p} uint32 words of fp32 tiles for `count` units and float64 inputs."""
    rng = np.random.default_rng(seed)
    one = np.zeros((1, TILE_WORDS), np.float32)
    if case == "softmax":
        # Scores; even units: one large score in row 0 (rows 0-3 move by more than
        # the lag), odd units: small scores (no row moves). l: positive sums.
        s = (rng.standard_normal((count, 32, 32 * STEP_WIDTH)) * 8).astype(np.float32)
        s[0::2, 0, 0] = 40.0
        s[1::2] *= np.float32(0.1)
        m = np.where(np.arange(32) < 4, -13.0, 8.0)
        l_old = rng.uniform(1, 5, 32).astype(np.float32).astype(np.float64)
        a = np.concatenate([tiles_of(s[u]) for u in range(count)])
        hand = tiles_of(np.broadcast_to((639.0 - m)[:, None], (32, 32)))
        return {
            "a": a.view(np.uint32),
            "b": one.view(np.uint32),
            "p": state_tile({0: m, 1: l_old}).view(np.uint32),
            "p_hand": hand.astype(np.float32).view(np.uint32),
            "s": s,
            "m": m,
            "l": l_old,
        }
    if case == "norm":
        x = rng.standard_normal((count * 32, 32 * WIDTH)).astype(np.float32)
        x[:32] *= 100.0  # one row of tiles with large values
        g = (1 + 0.1 * rng.standard_normal(32 * WIDTH)).astype(np.float32)
        a = np.concatenate([tiles_of(x[32 * u : 32 * u + 32]) for u in range(count)])
        mean = np.full((32, 32), 1.0 / (32 * WIDTH), np.float32)
        weight = np.broadcast_to(g, (32, 32 * WIDTH))
        eye = np.eye(32, dtype=np.float32)
        hand = np.concatenate([tiles_of(eye), tiles_of(mean), tiles_of(weight)])
        return {
            "a": a.view(np.uint32),
            "b": one.view(np.uint32),
            "p": p_tile(g).view(np.uint32),
            "p_hand": hand.view(np.uint32),
            "x": x,
            "g": g,
        }
    if case == "exp":
        # Scores; m each row's max over every unit (the state tile is one page).
        s = (rng.standard_normal((count * 32, 32)) * 8).astype(np.float32)
        m = s.reshape(count, 32, 32).max(axis=(0, 2)).astype(np.float64)
        a = np.concatenate([tiles_of(s[32 * u : 32 * u + 32]) for u in range(count)])
        lanes = np.ones((1, 32))
        state = (EXP_SCALE * m)[:, None] * lanes
        bias = (639 - np.float32(EXP_A) * m)[:, None] * lanes
        return {
            "a": a.view(np.uint32),
            "b": one.view(np.uint32),
            "p": tiles_of(state.astype(np.float32)).view(np.uint32),
            "p_hand": tiles_of(bias.astype(np.float32)).view(np.uint32),
            "m": m,
        }
    a = (rng.standard_normal((count, TILE_WORDS)) * 8).astype(np.float32)
    if count > 1:
        e = rng.uniform(-40, 40, TILE_WORDS)
        a[1] = (np.sign(rng.standard_normal(TILE_WORDS)) * 2.0**e).astype(np.float32)
    b = (rng.standard_normal((count, TILE_WORDS)) * 8).astype(np.float32)
    if INPUTS[case] == 1:
        b = one
    p = one.view(np.uint32)
    return {"a": a.view(np.uint32), "b": b.view(np.uint32), "p": p, "p_hand": p}


def softmax_reference(data):
    """float64 of the step per unit: P [32, 96], m, l, alpha [32], the word."""
    x = data["s"].astype(np.float64) * float(np.float32(EXP_A))
    m_old = data["m"][None, :]
    r = x.max(axis=2)
    m_new = np.where(r > m_old + STEP_LAG, r, m_old)
    alpha = np.exp2(m_old - m_new)
    p = np.exp2(x - m_new[:, :, None])
    l_new = alpha * data["l"][None, :] + p.sum(axis=2)
    return {
        "p": p,
        "m": m_new,
        "l": l_new,
        "alpha": alpha,
        "word": (m_new - m_old).max(1),
    }


def softmax_words(data, form):
    """The outputs a correct body would give (build mode's made-up results)."""
    ref = softmax_reference(data)
    out = []
    for u in range(len(data["s"])):
        out.append(tiles_of(ref["p"][u].astype(np.float32)))
        if form == "hand":
            v = np.broadcast_to((639.0 - ref["m"][u])[:, None], (32, 32))
            st = tiles_of(v.astype(np.float32))
            st[0, 257] = np.float32(ref["word"][u] > 0)
        else:
            st = state_tile({0: ref["m"][u], 1: ref["l"][u], 2: ref["alpha"][u]})
            st[0, WORD] = ref["word"][u]
        out.append(st)
    return np.concatenate(out).astype(np.float32).view(np.uint32)


def _rel(got, ref):
    """Max relative error of float64 `got` against `ref` (|ref| > 1e-30), and the
    count of non-finite values."""
    keep = np.abs(ref) > 1e-30
    rel = np.abs(got[keep] - ref[keep]) / np.abs(ref[keep])
    return float(rel.max()) if rel.size else 0.0, int((~np.isfinite(got)).sum())


def check_softmax(data, outputs, infos, nonces):
    """The record of the step: per body its cycles, P and m against float64, the
    emitted body's l, alpha and word too, the hand body's flag; the verdict."""
    ref = softmax_reference(data)
    rec = {"case": "softmax", "tolerance": STEP_TOLERANCE}
    count = len(data["s"])
    for body, info in infos.items():
        form = BODIES[body][1]
        got = _cycles(info, nonces[body])
        out = outputs[body].reshape(count, STEP_WIDTH + 1, TILE_WORDS)
        p = np.stack(
            [matrix_of(out[u, :STEP_WIDTH].view(np.float32)) for u in range(count)]
        ).astype(np.float64)
        errs, bad = {}, 0
        errs["p"], bad = _rel(p, ref["p"])
        if form == "hand":
            st = np.stack(
                [matrix_of(out[u, -1:].view(np.float32)) for u in range(count)]
            )
            m = 639.0 - st[:, :, 0].astype(np.float64)
            flag = out[:, -1].view(np.float32)[:, 257]
            errs["m_abs"] = float(np.abs(m - ref["m"]).max())
            got["flag_ok"] = bool(np.array_equal(flag > 0, ref["word"] > 0))
        else:
            slots = [
                [state_slot(out[u, -1], k) for k in range(3)] for u in range(count)
            ]
            spread = max(sp for row in slots for _, sp in row)
            m, l_new, alpha = (np.stack([row[k][0] for row in slots]) for k in range(3))
            errs["m"], _ = _rel(m, ref["m"])
            errs["l"], _ = _rel(l_new, ref["l"])
            errs["alpha"], _ = _rel(alpha, ref["alpha"])
            word = out[:, -1].view(np.float32)[:, WORD].astype(np.float64)
            got["word_ok"] = bool(np.array_equal(word > 0, ref["word"] > 0))
            got["word_rel"], _ = _rel(word, ref["word"])
            got["lane_spread"] = spread
        got["errors"], got["not_finite"] = errs, bad
        within = all(v <= STEP_TOLERANCE[k] for k, v in errs.items())
        flags = got.get("flag_ok", True) and got.get("word_ok", True)
        got["ok"] = (
            got["nonce_ok"] and got["error"] == 0 and not bad and within and flags
        )
        rec[form] = got
    return rec


def _cycles(info, nonce):
    """The nonce, error word and cycles (body, unit) of one body's info page."""
    got = {"nonce_ok": int(info[NONCE]) == nonce, "error": int(info[ERROR])}
    n = min(int(info[UNITS]), MAX_UNITS)
    if n:
        sync = statistics.median(int(v) for v in info[SYNC : SYNC + SYNCS])
        spans = [int(v) for v in info[BODY : BODY + n]]
        loops = [int(v) for v in info[LOOP : LOOP + n]]
        got["sync_span"] = sync
        got["body"] = {
            "min": min(spans),
            "median": statistics.median(spans),
            "max": max(spans),
            "less_sync": statistics.median(spans) - sync,
        }
        got["unit_median"] = statistics.median(loops[1:] or loops)
    return got


def reference(case, data):
    """float64 of the case's expression on the fp32 inputs, as the output's words."""
    a = data["a"].view(np.float32).astype(np.float64)
    b = data["b"].view(np.float32).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if case == "silu":
            return a / (1 + np.exp(-a))
        if case == "add":
            return a + b
        if case == "scale":
            return a * float(np.float32(SCALE))
        if case == "gate":
            return a / (1 + np.exp(-a)) * b
    if case == "exp":
        shift = tiles_of(np.broadcast_to(data["m"][:, None], (32, 32)))
        return np.exp(EXP_SCALE * (a - shift))
    x, g = data["x"].astype(np.float64), data["g"].astype(np.float64)
    y = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + EPS) * g
    count = len(x) // 32
    return np.concatenate([tiles_of(y[32 * u : 32 * u + 32]) for u in range(count)])


def errors(out, ref):
    """Max and mean relative error of fp32 words `out` against float64 `ref`."""
    y = out.view(np.float32).astype(np.float64)
    finite = np.isfinite(ref) & (np.abs(ref) > 1e-30)
    rel = np.abs(y[finite] - ref[finite]) / np.abs(ref[finite])
    bad = int((~np.isfinite(y[finite])).sum())
    return {"max": float(rel.max()), "mean": float(rel.mean()), "not_finite": bad}


def check(case, data, outputs, infos, nonces):
    """The record of one case: per body its nonce, error word, cycles of the body
    and of the whole unit, its error against float64 and the verdict."""
    if case == "softmax":
        return check_softmax(data, outputs, infos, nonces)
    ref = reference(case, data)
    rec = {"case": case, "tolerance": TOLERANCE[case]}
    hand = None
    for body, info in infos.items():
        form = BODIES[body][1]
        got = _cycles(info, nonces[body])
        got["vs_float64"] = errors(outputs[body], ref)
        bound = TOLERANCE[case]
        if form == "hand":
            bound = HAND_TOLERANCE.get(case, bound)
        elif form == "stated":
            bound = STATED_TOLERANCE[case]
        got["ok"] = (
            got["nonce_ok"]
            and got["error"] == 0
            and got["vs_float64"]["not_finite"] == 0
            and got["vs_float64"]["max"] <= bound
        )
        if form == "hand":
            hand = outputs[body]
        elif hand is not None:
            got["vs_hand"] = errors(outputs[body], hand.view(np.float32).astype(float))
            got["bits_equal_hand"] = bool(np.array_equal(outputs[body], hand))
        rec[form] = got
    return rec


def bodies_of(cases, card=False):
    out = [b for b, (case, _) in BODIES.items() if case in cases]
    return [b for b in out if not (card and b in ALREADY_RUN)]


def build(args):
    """Mock device: every body of `args.cases` built by the runtime's JIT, its size
    against the kernel buffer; the host checks run on made-up results."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(HERE.parents[1] / ".logs" / "jit-rules" / "mock")
    generate()
    sizes = {}
    with km.Session() as session:
        for body in bodies_of(args.cases):
            case, form = BODIES[body]
            data = inputs_of(case, args.units, args.seed)
            log(f"build {case} {form}")
            run, *_ = launch(session, body, data)
            _, *built = run.build()
            got = km_session.kernel_sizes(built)
            sizes[f"{case} {form}"] = [got.total, got.limit, got.text]
            log(f"built {case} {form}: kernel buffer {got.total} of {got.limit} B")
    for case in args.cases:
        data = inputs_of(case, args.units, args.seed)
        if case != "softmax":
            ref = reference(case, data).astype(np.float32).view(np.uint32)
        outputs, infos, nonces = {}, {}, {}
        for body in bodies_of([case]):
            if case == "softmax":
                ref = softmax_words(data, BODIES[body][1])
            info = np.zeros(TILE_WORDS, np.uint32)
            info[NONCE], info[UNITS] = 7, args.units
            info[SYNC : SYNC + SYNCS] = 10
            info[BODY : BODY + args.units] = 100 + body
            info[LOOP : LOOP + args.units] = 200 + body
            outputs[body], infos[body], nonces[body] = ref, info, 7
        log(json.dumps(check(case, data, outputs, infos, nonces)))
    print(json.dumps({"kernel_buffer": sizes}), flush=True)
    return 0


def card(args):
    """One core: each body of `args.cases` over the same inputs; values, cycles."""
    costs = generate()
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    started = time.time()
    clock, records, saved = [aiclk_mhz()], [], {}
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for case in args.cases:
            data = inputs_of(case, args.units, args.seed)
            outputs, infos, nonces = {}, {}, {}
            for body in bodies_of([case], card=True):
                if args.forms and BODIES[body][1] not in args.forms:
                    continue
                run, t_out, t_info, nonce = launch(session, body, data)
                log(f"launch {case} {BODIES[body][1]}")
                run.run()
                session.device.finish()
                log(f"done {case} {BODIES[body][1]}; reading back")
                outputs[body] = t_out.read().reshape(-1, TILE_WORDS).copy()
                infos[body] = t_info.read().reshape(-1).copy()
                nonces[body] = nonce
                clock.append(aiclk_mhz())
            rec = check(case, data, outputs, infos, nonces)
            saved.update({f"{case}_{BODIES[b][1]}": out for b, out in outputs.items()})
            saved.update({f"{case}_input_{k}": v for k, v in data.items()})
            unit = {"softmax": "softstep"}.get(case, case)
            rec["estimates"] = {k: v for k, v in costs.items() if k.startswith(unit)}
            log(json.dumps(rec))
            records.append(rec)
    if args.save:
        np.savez(args.save, **saved)
    out = {"aiclk_mhz": clock, "load": load, "wall_s": round(time.time() - started, 1)}
    print(json.dumps(out), flush=True)
    clean = all(
        r[form]["nonce_ok"] and r[form]["error"] == 0
        for r in records
        for form in FORMS
        if form in r
    )
    return 0 if clean else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--units", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", help="card: the outputs and inputs into this .npz file")
    p.add_argument("--forms", nargs="+", help="card: only these forms (hand, ...)")
    args = p.parse_args()
    if not 1 <= args.units <= MAX_UNITS:
        p.error(f"--units is 1 .. {MAX_UNITS}")
    if "norm" in args.cases and args.units * WIDTH > MAX_UNITS * WIDTH:
        p.error("too many norm rows")
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
