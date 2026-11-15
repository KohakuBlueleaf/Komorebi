"""Tile-rules experiment: bodies komorebi.dsl.tile emits against the hand bodies they
stand in for, on one core (README.md). `rules.py build`: a mock device builds every
body, the host checks run on made-up results; `rules.py card --cases ..`: values
against float64 and cycles of each body."""

import argparse
import hashlib
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
from komorebi.dsl.tile import emit, row_operand, trace, unit_program
from komorebi.dsl.tile.ir import BF16, F32, KernelIR, TensorParam
from komorebi.dsl.tile.matrix import product_program
from komorebi.dsl.tile.rows import row_program
from komorebi.dsl.tile.steps import SLOT, WORD, step_program
from komorebi.dsl.tile.unit import matrix_cost

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
GENERATED = KERNELS / "generated"
# The runtime's kernel cache: `build` fills it on the mock device, a card job reads it
# (the same build key on the card), so that no kernel is compiled inside a card job.
JIT_CACHE = HERE.parents[2] / ".logs" / "jit-rules" / "mock"
# rules.hpp's layout.
CB_A, CB_B, CB_P, CB_OUT, CB_INFO, MAX_UNITS = 0, 1, 2, 16, 24, 32
CB_SQUARES, CB_SCALE, CB_Y = 3, 4, 5  # the hand norm's
CB_KV, CB_Q, CB_PROB, CB_ST, CB_ONES = 0, 1, 2, 3, 4  # the attention body's
NONCE, ERROR, UNITS, BODY, LOOP, SYNC, SYNCS = 0, 1, 2, 64, 128, 192, 16
SUM_PART = 256  # BODY 20: per unit the products and the sum part (rules.hpp kSumPart)
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
    # the dense project's product: bf16 operands at HiFi2 (SrcA's top 5 bits in its
    # first phase); the emitted words are held to the library's bits
    "product": 2**-4,
    # a product in the body at HiFi4 (bf16 terms exact, the sum not one rounding:
    # about 1e-3 on the softmax step's sums), silu after it; the gate two of them
    "mm_silu": 2**-8,
    "mm_gate": 2**-7,
    # a per-row operand (.col()) into rsqrt and sigmoid: their constant registers
    # loaded in the per-row path (silu's 1e-6 for the exp and reciprocal)
    "rsqrt_col": 1e-6,
    "sigmoid_col": 1e-6,
}
MM_FIDELITY = "HiFi4"  # the tile function's default (km.tile_function)
MM_PRODUCT_TOLERANCE = 2**-8  # a @ w1 alone at HiFi4 (a band's rows past it)
PRODUCT_FIDELITY = "HiFi2"  # projects/dense's
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
# The fast exponent's clamp: unit 0 cycles through these t, the others take t in [-30,
# 0]; 2^t within the stated error where 2^t >= 2^-126, else within 2^-126 (finite).
CLAMP_ARGUMENTS = (-1, -64, -126, -127.5, -128, -150, -200, -300, -1e4)
CLAMP_FLOOR = 2.0**-126
ATTENTION = HERE.parents[2] / "projects" / "attention" / "kernels"
EXP_SCALE = 0.125  # 1 / sqrt(64), exact in fp32
EXP_A = EXP_SCALE * 1.4426950408889634  # the hand body's A = scale log2 e
EXP_TILE = {
    12: 3,
    13: 1,
    16: 1,
    21: 1,
    31: 3,
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
    18: ("attn", "hand"),
    19: ("attn", "emitted"),
    20: ("softmax", "ruled"),
    21: ("exp", "ruled"),
    22: ("product", "hand"),
    23: ("product", "emitted"),
    24: ("add", "band"),
    25: ("gate", "band"),
    26: ("norm", "band"),
    27: ("product", "trace"),
    28: ("mm_silu", "emitted"),
    29: ("mm_gate", "emitted"),
    30: ("mm_gate", "band"),
    31: ("clamp", "hand"),
    32: ("clamp", "emitted"),
    33: ("rsqrt_col", "emitted"),
    34: ("sigmoid_col", "emitted"),
}
FORMS = ("hand", "emitted", "macro", "stated", "accurate", "ruled", "band", "trace")
# Products inside the tile function (kernels/fused.cpp), one body each; BODY 27 is
# `a @ w` through the trace, BODY 23's words (checked by `build`), not run again.
FUSED = {27: "mm_hifi2", 28: "mm_silu", 29: "mm_gate", 30: "mm_gate_band"}
# The band's bodies (form "band"): the live rows a runtime argument, run at each of
# `card --rows`; their headers' names.
BAND = {24: "add_band", 25: "gate_band", 26: "norm_band", 30: "mm_gate_band"}
BAND_ROWS = {"add": 4, "gate": 4, "norm": 8, "mm_gate": 4}  # the rows a band runs in
CASES = ("silu", "add", "scale", "gate", "norm", "exp", "softmax", "attn", "product")
CASES += ("mm_silu", "mm_gate", "clamp", "rsqrt_col", "sigmoid_col")
# Run on the card already, over the same inputs (the tile-front experiment): built
# and checked here, not run again.
ALREADY_RUN = {0, 12, 27}
# Run on the card with the compiler before the rules of the vector-gap work (their
# headers now hold the rules' code; their numbers stay in the README): not run again.
EARLIER_CODE = {15, 16}
INPUTS = {
    **{"silu": 1, "add": 2, "scale": 1, "gate": 2, "norm": 1, "exp": 1},
    "softmax": 1,
    "attn": 1,
    "product": 2,
    "mm_silu": 2,
    "mm_gate": 3,
    "clamp": 1,
    "rsqrt_col": 2,
    "sigmoid_col": 2,
}
# The per-core attention body (kernels/attention.cpp): one query row of tiles, steps
# of 3 key tiles (--units steps), head and value dims of 2 tiles.
HEAD_TILES = VALUE_TILES = 2
# O / l against float64 (of the row's largest |O|): exp 1.75e-3, TF32 P and V 2^-11.
ATTN_TOLERANCE = 4e-3
ATTN_FIDELITY = "HiFi2"  # projects/attention/card.py's
# Words a step on the matrix unit: 16 MVMUL a tile product a phase (HiFi2: 2).
ATTN_WORDS = {"score": HEAD_TILES * 3 * 32, "pv": 3 * VALUE_TILES * 32, "sum": 3 * 32}


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


def norm_ir(name="norm"):
    """y = x rsqrt(mean(x^2) + eps) g over rows of WIDTH tiles, g one weight row."""
    x, g = TensorParam("x", False, F32), TensorParam("g", False, F32)
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [x, g, y], [], {})
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


def col_ir(name, fn):
    """y = fn(a, s.col()) over a stream of fp32 tiles, s each row's value in all its
    lanes (a per-row operand, the stream form's per-row path)."""
    a, s = TensorParam("a", False, F32), TensorParam("s", False, F32)
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [a, s, y], [], {})
    with trace.Tracer(ir):
        xs, ss, ys = (trace.TensorProxy(t) for t in (a, s, y))
        for t in trace.grid(trace.num_tiles(xs)):
            x, rows = trace.load(xs.tiles[t]), trace.load(ss.tiles[t])
            trace.store(ys.tiles[t], fn(x, rows.col()))
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
            alpha = trace.exp2(
                m_old - m_new, **exp, exact_at_zero=True
            )  # none if unmoved
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
    for name in ("add", "gate"):
        ir = stream(f"{name}_band", *fns[name])
        out[f"{name}_band"] = unit_program(ir, load_macro=True, rows=True)
    band = norm_ir("norm_band")
    out["norm_band"] = row_program(band, WIDTH, ROWS_TILE, P_TILE, rows=True)
    out["expstep"] = unit_program(expstep_ir())
    out["expstep_flash"] = unit_program(expstep_ir("expstep_flash", **STEP_EXP))
    exp2 = stream("exp2_flash", 1, lambda x: trace.exp2(x, **STEP_EXP))
    out["exp2_flash"] = unit_program(exp2)
    out["rsqrt_col"] = unit_program(
        col_ir("rsqrt_col", lambda a, s: a * trace.rsqrt(s))
    )
    sigmoid = col_ir("sigmoid_col", lambda a, s: trace.sigmoid(a + s))
    out["sigmoid_col"] = unit_program(sigmoid)
    out["softstep"] = step_program(softmax_ir(), STEP_WIDTH, STEP_TILE)
    accurate = softmax_ir("softstep_accurate", {})
    out["softstep_accurate"] = step_program(accurate, STEP_WIDTH, STEP_TILE)
    name = f"product_{PRODUCT_FIDELITY.lower()}"
    out[name] = product_program(name, PRODUCT_FIDELITY)
    out["mm_hifi2"] = unit_program(
        mm_ir("mm_hifi2", 2, lambda a, w: a @ w), fidelity=PRODUCT_FIDELITY
    )
    for name, inputs, fn, rows in (
        ("mm_silu", 2, lambda a, w: trace.silu(a @ w), False),
        ("mm_gate", 3, lambda a, w1, w3: trace.silu(a @ w1) * (a @ w3), False),
        ("mm_gate_band", 3, lambda a, w1, w3: trace.silu(a @ w1) * (a @ w3), True),
    ):
        ir = mm_ir(name, inputs, fn)
        out[name] = unit_program(ir, load_macro=True, rows=rows, fidelity=MM_FIDELITY)
    return out


def mm_ir(name, inputs, fn):
    """The IR of y = fn(a, w1[, w3]) over a stream of bf16 tiles, `@` a tile product."""
    params = [TensorParam(n, False, BF16) for n in ("a", "w1", "w3")[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(p) for p in params]
        ys = trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            loads = [trace.load(p.tiles[t]) for p in proxies]
            trace.store(ys.tiles[t], fn(*loads))
    return ir


def code_of(text, name):
    """The C++ lines of function `name` of an emitted header, without comments."""
    lines = text.split(f"inline void {name}(", 1)[1].split("\n    }", 1)[0]
    return [x for x in lines.splitlines()[1:] if not x.lstrip().startswith("//")]


def sources_hash():
    """The hash of the sources the headers come from (the tile package, this file)."""
    tile = Path(km.__file__).resolve().parent / "dsl" / "tile"
    digest = hashlib.sha256()
    for path in sorted(tile.glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generated_costs():
    """The costs `build` wrote beside the headers; a card job compiles nothing (the
    watcher kills a silent start). Refuses headers built from other sources."""
    try:
        saved = json.loads((GENERATED / "costs.json").read_text())
    except (OSError, ValueError):
        sys.exit("no generated headers: run `rules.py build` first, outside the job")
    if saved["sources"] != sources_hash():
        sys.exit("the generated headers are stale: run `rules.py build` again")
    log(f"estimates {json.dumps(saved['costs'])} (from build)")
    return saved["costs"]


def generate():
    """Writes every emitted header into kernels/generated (and their costs, for the
    card job); returns the units' costs; logs the load-macro rule's refusals."""
    GENERATED.mkdir(exist_ok=True)
    costs = {}
    made = units()
    for name, unit in made.items():
        (GENERATED / f"{name}.hpp").write_text(emit(unit))
        log(unit.dump())
        costs[name] = unit.cost("math")[1]
        if unit.band:
            costs[f"{name} 1 row"] = unit.cost("math", 1)[1]
        if unit.parts:
            parts, unit.parts = unit.parts, {}
            costs[f"{name} first part"] = unit.cost("math")[1]
            unit.parts = parts
    refused = {}
    for name, inputs, fn in (
        ("silu", 1, trace.silu),
        ("gate", 2, lambda a, b: trace.silu(a) * b),
    ):
        unit = unit_program(stream(name, inputs, fn), load_macro=True)
        refused[name] = next(p.done for p in unit.passes if p.name == "load_macro")
    log(f"estimates {json.dumps(costs)}")
    log(f"load-macro refusals {json.dumps(refused)}")
    library = f"product_{PRODUCT_FIDELITY.lower()}"
    same = code_of(emit(made["mm_hifi2"]), "mm_hifi2") == code_of(
        emit(made[library]), library
    )
    log(f"a @ w through the trace: the words of {library}: {same}")
    if not same:
        sys.exit("mm_hifi2 is not the product the card ran (BODY 23)")
    saved = {"sources": sources_hash(), "costs": costs}
    (GENERATED / "costs.json").write_text(json.dumps(saved))
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
    return {10: 2 + WIDTH, 11: 1, 24: 0, 25: 0, 26: 1, 32: 0, 33: 0, 34: 0}.get(
        body, int(body >= 12)
    )


def setting(body):
    """The compile-time setting of BODY `body`."""
    case, form = BODIES[body]
    unit_tiles = {"norm": WIDTH, "softmax": STEP_WIDTH}.get(case, 1)
    # BODY 20: B is the tile of ones for the row sums' products, not copied
    inputs = 2 if body == 20 else INPUTS[case]
    copy_b = inputs == 2 and (case, form) not in (
        ("gate", "hand"),
        ("softmax", "ruled"),
    )
    copy_b = copy_b and case != "product"
    return {
        "BODY": body,
        "UNIT_TILES": unit_tiles,
        "PACK_STATE": int(case == "softmax"),
        "STEP_THRESHOLD": f"{bits(639.0 + STEP_LAG)}u",
        "INPUTS": inputs,
        "COPY_B": int(copy_b),
        "HAS_P": 0 if case == "product" else p_pages(body),
        "P_TILE": STEP_TILE if case == "softmax" else EXP_TILE.get(body, P_TILE),
        "EXP_A": f"{float(np.float32(EXP_A))!r}f",
        "SCALE_BITS": f"{bits(SCALE)}u",
        "INV_N_BITS": f"{bits(1.0 / (32 * WIDTH))}u",
        "EPS_BITS": f"{bits(EPS)}u",
    }


def launch_attn(session, body, data, stage=3):
    """The per-core attention body (BODY 18, 19) on core (0, 0) over `data`; `stage`
    1, 2: only the score products (+ the hand step), packed out (attn_stages.py)."""
    steps = len(data["kv"]) // (3 * (HEAD_TILES + VALUE_TILES))
    columns = VALUE_TILES + (1 if BODIES[body][1] == "hand" else 0)
    nonce = secrets.randbits(32) | 1
    first = data["st_hand"] if BODIES[body][1] == "hand" else data["st"]

    def bf16_tiles(words):  # fp32 tiles in face order as a bf16 tile tensor
        tiles = words.view(np.float32)
        return session.tensor(
            np.concatenate([matrix_of(t[None]) for t in tiles]), "bfloat16", "tile"
        )

    tensors = [bf16_tiles(data["kv"]), bf16_tiles(data["q"])]
    tensors += [session.tensor(first, "uint32"), bf16_tiles(data["ones"])]
    t_out = session.tensor(np.zeros((columns + 1, TILE_WORDS), np.uint32), "uint32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core, page = [(0, 0, 0, 0)], 4 * TILE_WORDS
    block = 3 * (HEAD_TILES + VALUE_TILES)
    run = session.launch()
    for cb, pages, dtype in (
        (CB_KV, 2 * block, "bfloat16"),
        (CB_Q, HEAD_TILES, "bfloat16"),
        (CB_PROB, 6, "bfloat16"),
        (CB_ST, 2, "float32"),
        (CB_ONES, 1, "bfloat16"),
        (CB_OUT, columns + 1, "float32"),
        (CB_INFO, 1, "float32"),
    ):
        size = page if dtype == "float32" else page // 2
        run.cb(core, cb, dtype, size, pages * size)
    accessors = [w for t in tensors for w in t.accessor_args()]
    words = [t.address for t in tensors] + [steps, block, HEAD_TILES]
    run.kernel(
        str(KERNELS / "attn_reader.cpp"), core, 1, accessors, [], [], [(0, 0, words)]
    )
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, columns + 1]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], [(0, 0, words)])
    defines = {
        "BODY": body,
        "STAGE": stage,
        "HEAD_TILES": HEAD_TILES,
        "VALUE_TILES": VALUE_TILES,
        "STEP_THRESHOLD": f"{bits(639.0 + STEP_LAG)}u",
        "EXP_A": f"{float(np.float32(EXP_A))!r}f",
    }
    options = {
        "math_fidelity": ATTN_FIDELITY,
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": {CB_ST: "UnpackToDestFp32"},
    }
    run.kernel(
        str(KERNELS / "attention.cpp"),
        core,
        2,
        [],
        [(k, str(v)) for k, v in defines.items()],
        [str(GENERATED), str(ATTENTION)],
        [(0, 0, [nonce, steps])],
        **km_session.compute_config(options),
    )
    return run, t_out, t_info, nonce


def launch_product(session, body, data, rows=32):
    """The product (BODY 22, 23, 27) on core (0, 0): per unit one tile of A times one
    of B, bf16 operands as the dense project's, in `kernels/product.cpp`; the bodies
    with a product inside (28 to 30: one tile of A, W1 and W3 a unit) in
    `kernels/fused.cpp`, `rows` the band's live rows."""
    fused = body in FUSED and body != 27
    count = len(data["a"])
    nonce = secrets.randbits(32) | 1

    def bf16_tiles(words):  # fp32 tiles in face order as a bf16 tile tensor
        tiles = words.view(np.float32)
        return session.tensor(
            np.concatenate([matrix_of(t[None]) for t in tiles]), "bfloat16", "tile"
        )

    inputs = INPUTS[BODIES[body][0]]
    tensors = [bf16_tiles(data["a"]), bf16_tiles(data["b"])]
    if inputs == 3:
        tensors.append(bf16_tiles(data["p"]))
    else:
        tensors.append(session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32"))
    t_out = session.tensor(np.zeros((count, TILE_WORDS), np.uint32), "uint32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core, page = [(0, 0, 0, 0)], 4 * TILE_WORDS
    run = session.launch()
    held = count if fused else 2  # fused.cpp holds every unit's tiles before the first
    run.cb(core, CB_A, "bfloat16", page // 2, held * page // 2)
    run.cb(core, CB_B, "bfloat16", page // 2, held * page // 2)
    if inputs == 3:
        run.cb(core, CB_P, "bfloat16", page // 2, held * page // 2)
    run.cb(core, CB_OUT, "float32", page, 2 * page)
    run.cb(core, CB_INFO, "float32", page, page)
    accessors = [w for t in tensors for w in t.accessor_args()]
    words = [t.address for t in tensors] + [count, 1, inputs, 0, 0]
    run.kernel(str(KERNELS / "reader.cpp"), core, 1, accessors, [], [], [(0, 0, words)])
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, count]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], [(0, 0, words)])
    fidelity = MM_FIDELITY if fused else PRODUCT_FIDELITY
    options = {"math_fidelity": fidelity, "fp32_dest_acc_en": True}
    run.kernel(
        str(KERNELS / ("fused.cpp" if fused else "product.cpp")),
        core,
        2,
        [],
        [("BODY", str(body))],
        [str(GENERATED)],
        [(0, 0, [nonce, count, rows])],
        **km_session.compute_config(options),
    )
    return run, t_out, t_info, nonce


def launch(session, body, data, stage=None, rows=32):
    """One program on core (0, 0) running BODY `body` over `data` (a, b, p: uint32
    words of fp32 tiles); returns (the Launch, the output, the info tensor, nonce).
    `stage`: (number, folder) of headers ahead of kernels/generated (with a STAGE
    define); `rows`: the live rows a band body takes (a runtime argument)."""
    case, form = BODIES[body]
    if case == "attn":
        return launch_attn(session, body, data)
    if case in ("product", "mm_silu", "mm_gate"):
        return launch_product(session, body, data, rows)
    defines = setting(body)
    headers = [str(GENERATED), str(ATTENTION)]
    if stage is not None:
        defines["STAGE"] = stage[0]
        headers.insert(0, str(stage[1]))
    unit_tiles = defines["UNIT_TILES"]
    count = len(data["a"]) // unit_tiles
    nonce = secrets.randbits(32) | 1
    hand = form == "hand"
    keys = ("a_hand" if hand and "a_hand" in data else "a", "b")
    keys += ("p_hand" if hand else "p",)
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
    # The reader's input count is the kernel's (BODY 20 waits for a B tile a unit).
    words += [count, unit_tiles, defines["INPUTS"], defines["HAS_P"], int(body == 10)]
    run.kernel(str(KERNELS / "reader.cpp"), core, 1, accessors, [], [], args(words))
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, out_tiles]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], args(words))
    # Unpacked to Dst: the buffers copy_tile reads. Not the hand norm's P (the matrix
    # unit's operands: such a buffer unpacks Float32 into its source registers).
    unpack = {CB_A: "UnpackToDestFp32"}
    if defines["COPY_B"]:
        unpack[CB_B] = "UnpackToDestFp32"
    if body == 11 or 12 <= body < 24 or body in (26, 31):
        unpack[CB_P] = "UnpackToDestFp32"
    if body == 20:
        unpack[CB_SQUARES] = "UnpackToDestFp32"  # the state, back into Dst
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
        headers,
        args([nonce, count, rows]),
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
    """The per-column operand tile of weight row g (64 values), placed by the tile
    compiler (`row_operand`)."""
    return np.asarray(row_operand(g, WIDTH), np.float32).reshape(1, TILE_WORDS)


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


def bf16(values):
    """fp32 values rounded to bf16 (nearest even), as fp32."""
    u = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def attn_inputs(steps, seed):
    """The query row, the steps' K and V blocks (pages in the hand body's order),
    the ones tile (bf16 tiles as fp32 words) and the first state of each body."""
    rng = np.random.default_rng(seed)
    d = 32 * HEAD_TILES
    q = bf16(rng.standard_normal((32, d)))
    k = bf16(rng.standard_normal((96 * steps, d)))
    v = bf16(rng.standard_normal((96 * steps, 32 * VALUE_TILES)))
    pages = []
    for s in range(steps):
        keys = range(3 * s, 3 * s + 3)
        pages += [
            tiles_of(k[32 * j : 32 * j + 32])[h]
            for h in range(HEAD_TILES)
            for j in keys
        ]
        pages += [
            tiles_of(v[32 * j : 32 * j + 32])[c]
            for j in keys
            for c in range(VALUE_TILES)
        ]
    first = state_tile({0: np.full(32, -1e4), 1: np.zeros(32), 2: np.zeros(32)})
    return {
        "kv": np.stack(pages).view(np.uint32),
        "q": tiles_of(q).view(np.uint32),
        "ones": np.ones((1, TILE_WORDS), np.float32).view(np.uint32),
        "st": first.view(np.uint32),
        "st_hand": np.zeros((1, TILE_WORDS), np.uint32),
        "q64": q.astype(np.float64),
        "k64": k.astype(np.float64),
        "v64": v.astype(np.float64),
    }


def attn_reference(data):
    """float64 softmax(q k^T / sqrt(d)) v of the row, and whether any row's lagging
    reference moved after the first step (the bodies do not rescale O)."""
    s = data["q64"] @ data["k64"].T * EXP_SCALE
    p = np.exp(s - s.max(axis=1, keepdims=True))
    o = p @ data["v64"] / p.sum(axis=1, keepdims=True)
    a = s / EXP_SCALE * float(np.float32(EXP_A))
    first = a[:, :96].max(axis=1)
    moved = bool((a[:, 96:].max(axis=1, initial=-np.inf) > first + STEP_LAG).any())
    return o, moved


def attn_words(data, form):
    """The outputs a correct body would give (build mode's made-up results)."""
    o, _ = attn_reference(data)
    tiles = list(tiles_of(o.astype(np.float32)))
    if form == "hand":
        tiles.append(np.ones(TILE_WORDS, np.float32))
        tiles.append(np.zeros(TILE_WORDS, np.float32))
    else:
        tiles.append(state_tile({0: np.zeros(32), 1: np.ones(32)})[0])
    return np.stack(tiles).astype(np.float32).view(np.uint32)


def attn_estimates(costs):
    """A step's estimate per unit: the emitted vector step's cost, the matrix unit's
    words at one a cycle (6 or more Dst blocks rotating) and at 6 (none): the
    rotation inside the library's own word order is not known (stated)."""
    out = {"vector emitted": costs["softstep"]}
    for name, words in ATTN_WORDS.items():
        rotating = [("MVMUL", i % 8) for i in range(words)]
        alone = [("MVMUL", 0)] * words
        out[f"matrix {name}"] = [matrix_cost(rotating), matrix_cost(alone)]
    return out


def check_attn(data, outputs, infos, nonces):
    """The record of the per-core body: per body its cycles a step (score products,
    vector step, P V products, the step), O / l against float64, the verdict."""
    ref, moved = attn_reference(data)
    rec = {"case": "attn", "tolerance": ATTN_TOLERANCE, "rows_moved_later": moved}
    for body, info in infos.items():
        form = BODIES[body][1]
        got = _cycles(info, nonces[body])
        steps = min(int(info[UNITS]), MAX_UNITS)
        for name, at in (("score", 64), ("vector", 256), ("pv", 320), ("step", LOOP)):
            spans = [int(x) for x in info[at + 1 : at + steps]] or [int(info[at])]
            # Only the vector step sits between two syncs; the products run from issue.
            less = got.get("sync_span", 0) if name == "vector" else 0
            got[name] = statistics.median(spans) - less
        out = outputs[body].view(np.float32)
        o = matrix_of(out[:VALUE_TILES]).astype(np.float64)
        if form == "hand":
            l_row = matrix_of(out[VALUE_TILES : VALUE_TILES + 1])[:, 0].astype(float)
        else:
            l_row, _ = state_slot(outputs[body][VALUE_TILES], 1)
        y = o / l_row[:, None]
        err = np.abs(y - ref).max(axis=1) / np.abs(ref).max(axis=1)
        got["o_error"], got["not_finite"] = float(err.max()), int(
            (~np.isfinite(y)).sum()
        )
        got["ok"] = (
            got["nonce_ok"]
            and int(info[ERROR]) == 0
            and not got["not_finite"]
            and got["o_error"] <= ATTN_TOLERANCE
            and not moved
        )
        rec[form] = got
    return rec


def inputs_of(case, count, seed):
    """{a, b, p} uint32 words of fp32 tiles for `count` units and float64 inputs."""
    if case == "attn":
        return attn_inputs(count, seed)
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
        ones = np.ones((count, TILE_WORDS), np.float32)  # BODY 20's row-sum operand
        return {
            "a": a.view(np.uint32),
            "b": ones.view(np.uint32),
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
    if case == "product":
        # bf16 values (the operands' format) in [0.5, 1.5): no cancellation in a sum
        a = bf16(rng.uniform(0.5, 1.5, (count, TILE_WORDS)))
        b = bf16(rng.uniform(0.5, 1.5, (count, TILE_WORDS)))
        return {
            "a": a.view(np.uint32),
            "b": b.view(np.uint32),
            "p": one.view(np.uint32),
        }
    if case in ("mm_silu", "mm_gate"):
        # A in [0.5, 1.5), W1 and W3 in [0.5, 1.5) / 16 (bf16): the products about 0.5
        # to 4.5, where silu bends, with no cancellation in a sum
        a = bf16(rng.uniform(0.5, 1.5, (count, TILE_WORDS)))
        w1 = bf16(rng.uniform(0.5, 1.5, (count, TILE_WORDS)) / 16)
        w3 = bf16(rng.uniform(0.5, 1.5, (count, TILE_WORDS)) / 16)
        return {
            "a": a.view(np.uint32),
            "b": w1.view(np.uint32),
            "p": w3.view(np.uint32),
        }
    if case in ("rsqrt_col", "sigmoid_col"):
        # a; s one value a row (in all its lanes), positive for rsqrt
        a = (rng.standard_normal((count, TILE_WORDS)) * 3).astype(np.float32)
        lo, hi = (0.5, 4.0) if case == "rsqrt_col" else (-3.0, 3.0)
        rows = rng.uniform(lo, hi, (count, 32)).astype(np.float32)
        b = np.concatenate(
            [tiles_of(np.broadcast_to(r[:, None], (32, 32)).copy()) for r in rows]
        )
        return {
            "a": a.view(np.uint32),
            "b": b.view(np.uint32),
            "p": one.view(np.uint32),
            "p_hand": one.view(np.uint32),
        }
    if case == "clamp":
        # BODY 32 computes 2^a; BODY 31 (the hand body) 2^(A x + v - 639) with v = 639
        # in its state tile, x = a / A (fp32): the same arguments, up to x's rounding
        t = rng.uniform(-30, 0, (count, TILE_WORDS))
        lanes = np.arange(TILE_WORDS) // 32 + np.arange(TILE_WORDS) % 32
        t[0] = np.array(CLAMP_ARGUMENTS)[lanes % len(CLAMP_ARGUMENTS)]
        a = t.astype(np.float32)
        x = (a / np.float32(EXP_A)).astype(np.float32)
        bias = np.full((1, TILE_WORDS), 639.0, np.float32)
        return {
            "a": a.view(np.uint32),
            "a_hand": x.view(np.uint32),
            "b": one.view(np.uint32),
            "p": one.view(np.uint32),
            "p_hand": bias.view(np.uint32),
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
        if form == "ruled":
            n = min(int(info[UNITS]), MAX_UNITS)
            spans = [int(v) for v in info[SUM_PART : SUM_PART + n]]
            if spans:
                got["sum_part"] = {
                    "median": statistics.median(spans),
                    "with_first": statistics.median(spans) + got["body"]["less_sync"],
                }
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
        if case == "rsqrt_col":
            return a / np.sqrt(b)
        if case == "sigmoid_col":
            return 1 / (1 + np.exp(-(a + b)))
    if case == "product":
        return products(a, b)
    if case in ("mm_silu", "mm_gate"):
        p1 = products(a, b)
        silu = p1 / (1 + np.exp(-p1))
        if case == "mm_silu":
            return silu
        return silu * products(a, data["p"].view(np.float32).astype(np.float64))
    if case == "exp":
        shift = tiles_of(np.broadcast_to(data["m"][:, None], (32, 32)))
        return np.exp(EXP_SCALE * (a - shift))
    x, g = data["x"].astype(np.float64), data["g"].astype(np.float64)
    y = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + EPS) * g
    count = len(x) // 32
    return np.concatenate([tiles_of(y[32 * u : 32 * u + 32]) for u in range(count)])


def products(a, b):
    """float64 tile products a[u] @ b[u] of tiles in face order, in face order."""
    out = [
        tiles_of(matrix_of(a[u : u + 1]) @ matrix_of(b[u : u + 1]))[0]
        for u in range(len(a))
    ]
    return np.stack(out)


def errors(out, ref):
    """Max and mean relative error of fp32 words `out` against float64 `ref`."""
    y = out.view(np.float32).astype(np.float64)
    finite = np.isfinite(ref) & (np.abs(ref) > 1e-30)
    rel = np.abs(y[finite] - ref[finite]) / np.abs(ref[finite])
    bad = int((~np.isfinite(y[finite])).sum())
    return {"max": float(rel.max()), "mean": float(rel.mean()), "not_finite": bad}


def clamp_reference(data, form):
    """float64 2^t of each body's own fp32 arguments (the hand body's A x)."""
    if form == "hand":
        x = data["a_hand"].view(np.float32).astype(np.float64)
        return np.exp2(float(np.float32(EXP_A)) * x)
    return np.exp2(data["a"].view(np.float32).astype(np.float64))


def check_clamp(data, outputs, infos, nonces):
    """The clamp's record: per body unit 0's read-back at each argument of
    CLAMP_ARGUMENTS (its first lane), and over every unit the largest relative error
    where 2^t >= 2^-126, the largest absolute error below it, the non-finite count."""
    rec = {"case": "clamp", "tolerance": STEP_EXP["error"], "floor": CLAMP_FLOOR}
    lanes = np.arange(TILE_WORDS) // 32 + np.arange(TILE_WORDS) % 32
    for body, info in infos.items():
        form = BODIES[body][1]
        got = _cycles(info, nonces[body])
        y = outputs[body].view(np.float32).astype(np.float64)
        ref = clamp_reference(data, form)
        table = {}
        for k, t in enumerate(CLAMP_ARGUMENTS):
            w = int(np.flatnonzero(lanes % len(CLAMP_ARGUMENTS) == k)[0])
            table[f"{t:g}"] = {"got": float(y[0, w]), "want": float(ref[0, w])}
        got["read_back"] = table
        high = ref >= CLAMP_FLOOR
        with np.errstate(invalid="ignore", over="ignore"):
            rel = np.abs(y[high] - ref[high]) / ref[high]
            low = np.abs(y[~high] - ref[~high])
        bad = int((~np.isfinite(y)).sum())
        got["vs_float64"] = {
            "max": float(np.nanmax(rel)),
            "low_abs_max": float(np.nanmax(low)) if low.size else 0.0,
            "not_finite": bad,
        }
        got["ok"] = (
            got["nonce_ok"]
            and got["error"] == 0
            and bad == 0
            and got["vs_float64"]["max"] <= STEP_EXP["error"]
            and got["vs_float64"]["low_abs_max"] <= CLAMP_FLOOR
        )
        rec[form] = got
    return rec


def check(case, data, outputs, infos, nonces):
    """The record of one case: per body its nonce, error word, cycles of the body
    and of the whole unit, its error against float64 and the verdict."""
    if case == "clamp":
        return check_clamp(data, outputs, infos, nonces)
    if case == "softmax":
        return check_softmax(data, outputs, infos, nonces)
    if case == "attn":
        return check_attn(data, outputs, infos, nonces)
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
        elif form in ("stated", "ruled"):
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


def face_rows():
    """The tile row of each of a tile's 1024 words in face order."""
    w = np.arange(TILE_WORDS)
    return w // 512 * 16 + w % 256 // 16


def check_band(case, data, out, info, nonce, rows):
    """The record of a band body at `rows` live rows: its cycles; on the rows of the
    live bands its error against float64 (the case's tolerance); past them, whether
    the output is the input a bit for bit (rows past the band are left as they were)."""
    got = _cycles(info, nonce)
    band = BAND_ROWS[case]
    end = min(32, -(-rows // band) * band) if 1 <= rows <= 32 else 32
    live = face_rows() < end
    ref = reference(case, data)
    got["rows"], got["band_end"] = rows, end
    got["vs_float64"] = errors(out[:, live], ref[:, live])
    if case == "mm_gate":
        # past the band Dst tile 0 holds a @ w1, the product the vector code reads
        a, w1 = (data[k].view(np.float32).astype(np.float64) for k in ("a", "b"))
        past = errors(out[:, ~live], products(a, w1)[:, ~live]) if end < 32 else None
        got["past_band_vs_product"] = past
        got["past_band_is_input"] = past is None or past["max"] <= MM_PRODUCT_TOLERANCE
    else:
        got["past_band_is_input"] = bool(
            np.array_equal(out[:, ~live], data["a"][:, ~live])
        )
    got["ok"] = (
        got["nonce_ok"]
        and got["error"] == 0
        and got["vs_float64"]["not_finite"] == 0
        and got["vs_float64"]["max"] <= TOLERANCE[case]
        and got["past_band_is_input"]
    )
    return got


def bodies_of(cases, card=False):
    out = [b for b, (case, _) in BODIES.items() if case in cases]
    return [b for b in out if not (card and b in ALREADY_RUN | EARLIER_CODE)]


def build(args):
    """Mock device: every body of `args.cases` built by the runtime's JIT, its size
    against the kernel buffer; the host checks run on made-up results."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
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
        if case not in ("softmax", "attn", "clamp"):
            ref = reference(case, data).astype(np.float32).view(np.uint32)
        outputs, infos, nonces = {}, {}, {}
        for body in bodies_of([case]):
            if case == "clamp":
                ref = clamp_reference(data, BODIES[body][1])
                ref = ref.astype(np.float32).view(np.uint32)
            if case == "softmax":
                ref = softmax_words(data, BODIES[body][1])
            if case == "attn":
                ref = attn_words(data, BODIES[body][1])
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
    costs = generated_costs()
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    log(f"kernels from the build's cache {JIT_CACHE}")
    with open("/proc/loadavg") as f:
        load = f.read().split()[:3]
    started = time.time()
    clock, records, saved = [aiclk_mhz()], [], {}
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for case in args.cases:
            data = inputs_of(case, args.units, args.seed)
            outputs, infos, nonces = {}, {}, {}
            bands = []
            for body in bodies_of([case], card=True):
                if args.forms and BODIES[body][1] not in args.forms:
                    continue
                if args.bodies and body not in args.bodies:
                    continue
                if body in BAND:
                    for rows in args.rows:
                        run, t_out, t_info, nonce = launch(
                            session, body, data, None, rows
                        )
                        log(f"launch {case} band, {rows} rows")
                        run.run()
                        session.device.finish()
                        out = t_out.read().reshape(-1, TILE_WORDS).copy()
                        info = t_info.read().reshape(-1).copy()
                        bands.append(check_band(case, data, out, info, nonce, rows))
                        log(f"band {case} {json.dumps(bands[-1])}")
                        saved[f"{case}_band_{rows}"] = out
                        clock.append(aiclk_mhz())
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
            if bands:
                rec["band"] = bands
            saved.update({f"{case}_{BODIES[b][1]}": out for b, out in outputs.items()})
            saved.update({f"{case}_input_{k}": v for k, v in data.items()})
            unit = {"softmax": "softstep", "clamp": "exp2_flash"}.get(case, case)
            rec["estimates"] = {k: v for k, v in costs.items() if k.startswith(unit)}
            if case == "attn":
                rec["estimates"] = attn_estimates(costs)
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
        if form in r and form != "band"
    ) and all(
        b["nonce_ok"] and b["error"] == 0 for r in records for b in r.get("band", [])
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
    p.add_argument(
        "--bodies", nargs="+", type=int, help="card: only these BODY numbers"
    )
    p.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[1, 32],
        help="card: the live rows each band body runs at",
    )
    args = p.parse_args()
    if not 1 <= args.units <= MAX_UNITS:
        p.error(f"--units is 1 .. {MAX_UNITS}")
    if "norm" in args.cases and args.units * WIDTH > MAX_UNITS * WIDTH:
        p.error("too many norm rows")
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
