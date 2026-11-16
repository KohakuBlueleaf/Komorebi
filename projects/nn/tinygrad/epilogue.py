"""tinygrad's kernels against the tile compiler: each elementwise part a kernel stores
(schedule.py's epilogues) as a tile function and its cost, or what refuses it; and one
of them, model A's first kernel's, as a program of its own on one core of the card.
Reads what schedule.py wrote; imports no tinygrad."""

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import session as km_session
from komorebi.bench import power
from komorebi.device import native
from komorebi.dsl.tile import emit, trace, unit_program
from komorebi.dsl.tile.ir import F32, KernelIR, PlanError, TensorParam
from komorebi.dsl.tile.unit import UnitError

HERE = Path(__file__).resolve().parent
KERNELS = HERE / "kernels"
GENERATED = KERNELS / "generated"
SCHEDULE = HERE / "results" / "schedule.json"
# The runtime's kernel cache: `build` fills it on the mock device, the card job reads
# it (the same build key), so that no kernel is compiled inside a card job.
JIT_CACHE = HERE.parents[2] / ".logs" / "jit-nn-tinygrad" / "mock"
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
CB_A, CB_B, CB_OUT, CB_INFO = 0, 1, 16, 24  # kernels/layout.hpp
NONCE, ERROR, TILES, STREAM, SYNC, SYNCS, BODY = 0, 1, 2, 4, 16, 16, 256
TILE_WORDS = 1024
LOG2E = 1.4426950408889634
# The tile compiler's gate against float64 (experiments/language/tile_rules).
TOLERANCE = 1e-6
CALLS = 4  # runs of the kept program a job: the first loads it, the rest are calls


def sigmoid_of(e):
    """x when `e` is tinygrad's sigmoid of x, 1 / (1 + 2^(x * -log2 e)), else None."""
    if e[0] != "recip" or e[1][0] != "add" or e[1][1] != ["const", 1.0]:
        return None
    power = e[1][2]
    if power[0] != "exp2" or power[1][0] != "mul" or power[1][2][0] != "const":
        return None
    return power[1][1] if math.isclose(power[1][2][1], -LOG2E) else None


def tile_source(e):
    """The tile function's expression (Python over the operands a, b, c, ..) for
    tinygrad's expression `e`: its sigmoid and x * sigmoid(x) become trace.sigmoid
    and trace.silu (tinygrad writes them out); raises KeyError on a name it lacks."""
    kind = e[0]
    if kind == "in":
        return "abcdefgh"[e[1]]
    if kind == "const":
        return repr(e[1])
    x = sigmoid_of(e)
    if x is not None:
        return f"trace.sigmoid({tile_source(x)})"
    if kind == "mul":
        for i in (1, 2):
            x = sigmoid_of(e[3 - i])
            if x is not None and x == e[i]:
                return f"trace.silu({tile_source(x)})"
        return f"({tile_source(e[1])} * {tile_source(e[2])})"
    if kind == "add":
        return f"({tile_source(e[1])} + {tile_source(e[2])})"
    if kind == "neg":
        return f"(-{tile_source(e[1])})"
    names = {"exp2": "trace.exp2", "sqrt": "trace.sqrt", "recip": "trace.reciprocal"}
    if kind in names:
        return f"{names[kind]}({tile_source(e[1])})"
    raise KeyError(kind)


def stream_ir(name, inputs, source):
    """The IR of y = source(a, b, ..) over a stream of fp32 tiles."""
    params = [TensorParam(n, False, F32) for n in "abcdefgh"[:inputs]]
    y = TensorParam("y", True, F32)
    ir = KernelIR(name, [*params, y], [], {})
    fn = eval(f"lambda {', '.join('abcdefgh'[:inputs])}: {source}", {"trace": trace})
    with trace.Tracer(ir):
        proxies = [trace.TensorProxy(p) for p in params]
        ys = trace.TensorProxy(y)
        for t in trace.grid(trace.num_tiles(proxies[0])):
            loads = [trace.load(p.tiles[t]) for p in proxies]
            trace.store(ys.tiles[t], fn(*loads))
    return ir


def render(epilogue, name="epilogue"):
    """{source, cost} of an epilogue's tile function, or {refused: why}."""
    try:
        source = tile_source(epilogue["expression"])
    except KeyError as kind:
        return {"refused": f"no tile function for tinygrad's {kind.args[0]}"}
    if not source.startswith(("trace", "(")):
        return {"source": source, "refused": "a store of an operand: nothing to render"}
    try:
        unit = unit_program(stream_ir(name, len(epilogue["inputs"]), source))
    except (PlanError, UnitError) as error:
        return {"source": source, "refused": str(error).split(" (at ")[0]}
    return {"source": source, "cost": unit.cost("math")[1]}


def renders(args):
    """Each kernel's stored expressions of the schedule, rendered or refused."""
    if not SCHEDULE.is_file():
        sys.exit(f"{SCHEDULE} is missing: `schedule.py --json {SCHEDULE}` writes it")
    schedule = json.loads(SCHEDULE.read_text())
    for model, items in schedule.items():
        for i, item in enumerate(items):
            for e in item.get("epilogue", []):
                got = render(e)
                print(
                    f"{model} {i:2d} {item['name']:24s} {json.dumps(got)}", flush=True
                )
    return 0


def gate_unit(case):
    """Model A's first kernel's epilogue (from the case file) as a unit program."""
    epilogue = json.loads(str(case["epilogue"]))
    got = render(epilogue)
    if "cost" not in got:
        sys.exit(f"model A's first epilogue does not render: {got}")
    source = got["source"]
    return source, unit_program(stream_ir("epilogue", len(epilogue["inputs"]), source))


def sources_hash():
    """The hash of what the header comes from (the tile package, this file)."""
    tile = Path(km.__file__).resolve().parent / "dsl" / "tile"
    digest = hashlib.sha256()
    for path in sorted(tile.glob("*.py")) + [Path(__file__).resolve()]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def words_of(values):
    """fp32 values as [tiles, 1024] uint32 words (the stream is elementwise: a tile is
    any 1024 of them, here in order)."""
    return (
        np.ascontiguousarray(values, np.float32).reshape(-1, TILE_WORDS).view(np.uint32)
    )


def launch(session, a, b):
    """The program on core (0, 0) over operand tiles a and b; (Launch, out, info,
    nonce)."""
    tiles = len(a)
    nonce = secrets.randbits(32) | 1
    t_a, t_b = session.tensor(a, "uint32"), session.tensor(b, "uint32")
    t_out = session.tensor(np.zeros((tiles, TILE_WORDS), np.uint32), "uint32")
    t_info = session.tensor(np.zeros((1, TILE_WORDS), np.uint32), "uint32")
    core = [(0, 0, 0, 0)]
    page = 4 * TILE_WORDS
    run = session.launch()
    for cb in (CB_A, CB_B, CB_OUT):
        run.cb(core, cb, "float32", page, 2 * page)
    run.cb(core, CB_INFO, "float32", page, page)
    accessors = t_a.accessor_args() + t_b.accessor_args()
    words = [t_a.address, t_b.address, tiles]
    run.kernel(str(KERNELS / "reader.cpp"), core, 1, accessors, [], [], [(0, 0, words)])
    accessors = t_out.accessor_args() + t_info.accessor_args()
    words = [t_out.address, t_info.address, tiles]
    run.kernel(str(KERNELS / "writer.cpp"), core, 0, accessors, [], [], [(0, 0, words)])
    options = {
        "math_fidelity": "HiFi4",
        "fp32_dest_acc_en": True,
        "unpack_to_dest_mode": {CB_A: "UnpackToDestFp32", CB_B: "UnpackToDestFp32"},
    }
    run.kernel(
        str(KERNELS / "compute.cpp"),
        core,
        2,
        [],
        [],
        [str(GENERATED)],
        [(0, 0, [nonce, tiles])],
        **km_session.compute_config(options),
    )
    return run, t_out, t_info, nonce


def relative(got, ref):
    """Max and mean relative error of fp32 `got` against `ref` (float64)."""
    y = got.astype(np.float64)
    finite = np.isfinite(ref) & (np.abs(ref) > 1e-30)
    rel = np.abs(y[finite] - ref[finite]) / np.abs(ref[finite])
    bad = int((~np.isfinite(y[finite])).sum())
    return {"max": float(rel.max()), "mean": float(rel.mean()), "not_finite": bad}


def check(case, out, info, nonce):
    """The run's record: nonce, error word, cycles, errors against float64 and
    against tinygrad's own output."""
    a, b = (case[k].reshape(-1).astype(np.float64) for k in "ab")
    ref = a / (1 + np.exp(-a)) * b
    got = out.view(np.float32).reshape(-1)
    tiles = int(info[TILES])
    sync = statistics.median(int(v) for v in info[SYNC : SYNC + SYNCS])
    spans = [int(v) for v in info[BODY : BODY + tiles]]
    rec = {"nonce_ok": int(info[NONCE]) == nonce, "error": int(info[ERROR])}
    rec["body_less_sync"] = statistics.median(spans) - sync if spans else None
    rec["body"] = [min(spans), statistics.median(spans), max(spans)] if spans else None
    rec["stream_cycles"] = int(info[STREAM])
    rec["vs_float64"] = relative(got, ref)
    rec["vs_tinygrad"] = relative(got, case["g"].reshape(-1).astype(np.float64))
    rec["tinygrad_vs_float64"] = relative(case["g"].reshape(-1), ref)
    rec["ok"] = (
        rec["nonce_ok"]
        and rec["error"] == 0
        and rec["vs_float64"]["not_finite"] == 0
        and rec["vs_float64"]["max"] <= TOLERANCE
        and rec["vs_tinygrad"]["max"] <= TOLERANCE
    )
    return rec


def build(args):
    """Mock device: the header emitted, the program built by the runtime's JIT into the
    cache; the host check on a made-up result (float64's, as fp32)."""
    case = np.load(args.case)
    source, unit = gate_unit(case)
    GENERATED.mkdir(exist_ok=True)
    (GENERATED / "epilogue.hpp").write_text(emit(unit))
    cost = unit.cost("math")[1]
    saved = {"sources": sources_hash(), "source": source, "cost": cost}
    (GENERATED / "costs.json").write_text(json.dumps(saved))
    print(f"tile function {source}: {cost} cycles a tile (the cost model)", flush=True)
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(args.cache)
    a, b = words_of(case["a"]), words_of(case["b"])
    with km.Session() as session:
        run, *_ = launch(session, a, b)
        _, *built = run.build()
        got = km_session.kernel_sizes(built)
        print(f"built: kernel buffer {got.total} of {got.limit} B", flush=True)
    av, bv = (case[k].astype(np.float64) for k in "ab")
    made = (av / (1 + np.exp(-av)) * bv).astype(np.float32)
    info = np.zeros(TILE_WORDS, np.uint32)
    info[NONCE], info[TILES], info[SYNC : SYNC + SYNCS] = 7, len(a), 10
    info[BODY : BODY + len(a)] = 100
    print(json.dumps(power.clocked(check(case, words_of(made), info, 7))), flush=True)
    return 0


def card(args):
    """One core: the kept program run CALLS times over the case; values, cycles, the
    host's time a call (run to finish)."""
    case = np.load(args.case)
    saved = json.loads((GENERATED / "costs.json").read_text())
    if saved["sources"] != sources_hash():
        sys.exit("the generated header is stale: run `epilogue.py build` again")
    print(f"tile function {saved['source']}: {saved['cost']} (from build)", flush=True)
    os.environ["TT_METAL_CACHE"] = str(args.cache)
    a, b = words_of(case["a"]), words_of(case["b"])
    started = time.time()
    records, calls = [], []
    with km.Session() as session:
        print(f"device open {time.time() - started:.1f}s", flush=True)
        run, t_out, t_info, nonce = launch(session, a, b)
        for i in range(CALLS):
            t_info.buffer.write(np.zeros(4 * TILE_WORDS, np.uint8))
            t_out.buffer.write(np.zeros(a.size * 4, np.uint8))
            start = time.perf_counter()
            run.run()
            session.device.finish()
            calls.append(round((time.perf_counter() - start) * 1e6, 1))
            out = t_out.read().reshape(-1, TILE_WORDS).copy()
            info = t_info.read().reshape(-1).copy()
            records.append(check(case, out, info, nonce))
            print(f"call {i}: {calls[-1]} us {json.dumps(records[-1])}", flush=True)
    clock = km_session.aiclk_mhz()
    out = {
        "calls_us": calls,
        "aiclk_mhz": clock,
        "wall_s": round(time.time() - started),
    }
    out["cost_model"] = saved["cost"]
    print(json.dumps(power.clocked(out)), flush=True)
    return 0 if all(r["ok"] for r in records) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["renders", "build", "card"])
    p.add_argument("--case", help="build, card: schedule.py's --gate file (.npz)")
    p.add_argument("--cache", type=Path, default=JIT_CACHE, help="the kernel cache")
    args = p.parse_args()
    if args.what == "renders":
        return renders(args)
    if args.case is None:
        p.error("build and card take --case")
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
