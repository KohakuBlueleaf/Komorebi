"""The same programs on one island and on many: attention (the class, and the attention
project's program with its plan over the islands), the SwiGLU class and the decode
layer class, each run on one session shape and its output words saved (--save) or
compared with a saved one-island run (--against). One JSON line; README.md."""

import argparse
import importlib
import json
import sys
import time
import traceback

import numpy as np

import komorebi as km
from experiments.language.attention_class import attention as ac
from experiments.language.decode_class import across
from experiments.language.decode_class import decode_layer as dl
from experiments.language.decode_class import run_layer as rl
from komorebi import islands
from komorebi.bench import power
from projects.attention import card
from projects.attention import plan as pl
from projects.decode import model as md

sw = importlib.import_module("examples.31_swiglu.swiglu_class")  # a digit first
STARTED = time.monotonic()
SHAPES = {  # the session each shape opens
    "one": {"card": 0},
    "two": {"cards": [0, 1]},
    "line3": {"card": 0, "islands": 3, "graph": islands.line_pairs(3)},
    "ring4": {"card": 0, "islands": 4, "graph": islands.ring_pairs(4)},
    "grid4": {"card": 0, "islands": 4, "graph": islands.grid_pairs(2, 2, wrap=True)},
}
PROGRAMS = ("attention", "attention_islands", "swiglu", "decode")
SEED = 1


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=tuple(SHAPES), default="one")
    parser.add_argument("--programs", default=",".join(PROGRAMS))
    parser.add_argument("--links", type=int, help="two cards: at most this many links")
    parser.add_argument("--kernel-bytes", type=int, help="the session's kernel buffer")
    parser.add_argument("--attention", default="t1", help="the attention shape")
    parser.add_argument("--calls", type=int, default=20, help="timed commands")
    parser.add_argument("--save", help="the outputs' words to this .npz")
    parser.add_argument("--against", help="compare with this .npz (a one-island run)")
    return parser.parse_args()


def islands_of(session):
    """The session's graph, or None on one card."""
    return getattr(session, "graph", None)


def edge(session):
    """The first rank of island 1 (a group placed around it crosses), else 60."""
    graph = islands_of(session)
    return graph.first(1) if graph else 60


def pages(session, tensor):
    """A tensor's pages as bytes, read while its run may still be loaded."""
    (rows,) = session.read(tensor)
    return np.array(rows, copy=True).reshape(-1).view(np.uint8)


def median(times):
    return round(float(np.median(times)), 2) if len(times) else None


def done(events):
    """Each event's microseconds; an event not done is an error naming it."""
    events = events if isinstance(events, list) else [events]
    for event in events:
        if event.kind != "done":
            raise km.KomorebiError(f"command not done: {event}")
    return [event.microseconds for event in events]


def attention_inputs(name):
    return ac.inputs(name, SEED)


def attention_class(session, args):
    """The class of experiments/language/attention_class on the session's grid: one
    ATTEND over every job's core, O's words."""
    call, causal, scale, q, k, v = attention_inputs(args.attention)
    plan = pl.make(call, session.grid)
    zeros = np.zeros((*q.shape[:-1], v.shape[-1]), np.float32)
    t = {
        "q": card.bf16_tensor(session, q),
        "k": card.bf16_tensor(session, k),
        "v": card.bf16_tensor(session, v),
        "o": card.bf16_tensor(session, zeros),
        "tables": card.table_tensor(session, pl.tables(plan)),
    }
    scale_bits, threshold_bits = pl.exp_constants(scale)
    defines = {"FLASH_SCALE_BITS": scale_bits, "FLASH_THRESHOLD_BITS": threshold_bits}
    if causal:
        defines["FLASH_CAUSAL"] = 1
    program = ac.Attention(
        q=t["q"], k=t["k"], v=t["v"], o=t["o"], tables=t["tables"],
        defines=defines, session=session, log=log,
    )  # fmt: skip
    run = session.load(program)
    try:
        group = (0, len(plan.jobs))
        words = (
            t["q"], t["k"], t["v"], t["o"], t["tables"], pl.K["kModeFull"], 1, 0, 0,
        )  # fmt: skip
        us = ac.timed(run, group, words, args.calls)
    finally:
        run.stop()
    return {"o": pages(session, t["o"])}, {"us": median(us), "group": list(group)}


def attention_islands(session, args):
    """The attention project's program (projects/attention, its README's --cards):
    on islands its plan over the islands as one grid (card.island_plan)."""
    call, causal, scale, q, k, v = attention_inputs(args.attention)
    grid = None
    if islands_of(session):
        plan, grid = card.island_plan(session, call)
    else:
        plan = pl.make(call, session.nocs().grid)
    zeros = np.zeros((*q.shape[:-1], v.shape[-1]), np.float32)
    t = {
        "q": card.bf16_tensor(session, q),
        "k": card.bf16_tensor(session, k),
        "v": card.bf16_tensor(session, v),
        "o": card.bf16_tensor(session, zeros),
        "tables": card.table_tensor(session, pl.tables(plan)),
    }
    attention = card.Attention(session, plan, t, scale, grid=grid)
    try:
        _, us = attention.calls(args.calls)
    finally:
        stopped = attention.stop()
    if not stopped["ok"]:
        raise km.KomorebiError(f"the run: {stopped}")
    return {"o": pages(session, t["o"])}, {"us": median(us), "cores": len(plan.jobs)}


def swiglu(session, args):
    """The SwiGLU class (examples/31_swiglu): a command a step over a group around
    island 1's first rank, then its chain as written (cores 0..11), its tensors held
    by the last island."""
    data = sw.inputs()
    t = sw.tensors_of(session, data, empty=False)
    graph = islands_of(session)
    held = {}
    if graph:
        last = len(graph.islands) - 1
        held = {"split": 0, "group": (graph.first(last), graph.count(last))}
    far = {n: session.tensor(m, "bfloat16", "tile", **held) for n, m in data.items()}
    far["hidden"] = session.empty((32, 96 * 32), "bfloat16", "tile", **held)
    far["y"] = session.empty((32, 32 * 32), "float32", "tile", **held)
    program = sw.SwiGLU(log=log, session=session, **t)
    up, down = (edge(session) - 6, 12), (edge(session) - 2, 4)
    run = session.load(program)
    try:
        steps = []
        for _ in range(1 + sw.RUNS):
            first = run.wait(
                run.submit(
                    sw.SwiGLU.up, up, args=(t["x"], t["w1"], t["w3"], t["hidden"])
                )
            )
            second = run.wait(
                run.submit(sw.SwiGLU.down, down, args=(t["hidden"], t["w2"], t["y"]))
            )
            steps.append(sum(done([first, second])))
        out = {"hidden": pages(session, t["hidden"]), "y": pages(session, t["y"])}
        names = ("x", "w1", "w3", "w2", "hidden", "y")
        layer = run.chain(program.forward(*(far[n] for n in names)))
        chain = [
            done(run.wait(run.submit(layer)))[0] for _ in range(1 + sw.RUNS)
        ]  # fmt: skip
        out["chain_hidden"] = pages(session, far["hidden"])
        out["chain_y"] = pages(session, far["y"])
        layer.free()
    finally:
        run.stop()
    extra = {"steps_us": median(steps[1:]), "chain_us": median(chain[1:])}
    extra |= {"up": list(up), "down": list(down)}
    return out, extra


def decode(session, args):
    """The decode layer class (experiments/language/decode_class/across.py): its
    chain, then the same five calls as a command a step (--steps-only), at the first
    ranks across.py gives (one card 0,0,0; two cards 6,103,70)."""
    graph = islands_of(session)
    firsts = (6, 103, 70) if graph and len(session.cards) > 1 else (0, 0, 0)
    qkv_first, o_first, up_first = firsts
    position = dl.CONTEXT - 1
    start = md.start(SEED, dl.MODEL)
    weights = md.layer_weights(dl.MODEL, SEED, 0)
    cache = md.cache(dl.MODEL, SEED, 0, dl.CONTEXT)
    t = across.tensors_of(session, weights, cache, graph is not None)
    rl.start_class(session, t, start)
    program = across.DecodeLayerAcross(session=session, log=log, **rl.examples(t))
    run = session.load(program, wait=1.0, deadline=0.5)
    ends = [(qkv_first, dl.QKV_CORES), (o_first, dl.O_CORES)]
    ends.append((up_first, dl.GATE_UP_CORES))
    low = min(f for f, _ in ends)
    high = max(f + n for f, n in ends)
    words = [t[n] for n in rl.CHAIN_NAMES]
    try:
        chain = run.chain(
            program.placed(*words, position, qkv_first, o_first, up_first),
            cores=(low, high - low),
        )
        times = [done(run.wait(run.submit(chain), timeout=10.0))[0]]
        out = {f"chain_{n}": pages(session, t[n]) for n in across.NAMES}  # one layer
        times += [done(run.wait(run.submit(chain), timeout=10.0))[0] for _ in range(10)]
        chain.free()
        rl.start_class(session, t, start)
        first = {"qkv": qkv_first, "gate_up": up_first}
        steps = [
            (handler, (first.get(step, o_first), cores), step_words)
            for step, handler, cores, step_words in rl.calls(program, t, position)
        ]
        stepped = [across.by_steps(run, steps)]
        out |= {f"steps_{n}": pages(session, t[n]) for n in across.NAMES}
        stepped += [across.by_steps(run, steps) for _ in range(10)]
    finally:
        run.stop()
    extra = {"chain_us": median(times[1:]), "steps_us": median(stepped[1:])}
    extra["firsts"] = list(firsts)
    return out, extra


RUNNERS = {
    "attention": attention_class,
    "attention_islands": attention_islands,
    "swiglu": swiglu,
    "decode": decode,
}
FLOAT32 = {"y", "chain_y"}  # the SwiGLU's y is float32; every other output bf16


def values(name, raw):
    """An output's bytes as float64 values (bf16 or float32 by name)."""
    if name.split(":")[-1] in FLOAT32:
        return raw.view(np.float32).astype(np.float64)
    return (
        (raw.view(np.uint16).astype(np.uint32) << 16)
        .view(np.float32)
        .astype(np.float64)
    )


def tracked(session):
    """Every tensor the session makes from now on, listed (each program's tensors are
    freed after it: an island session's table holds 32 at once)."""
    made = {}

    def keeping(made_by):
        def making(*args, **options):
            tensor = made_by(*args, **options)
            made[id(tensor)] = tensor
            return tensor

        return making

    for name in ("tensor", "empty"):
        setattr(session, name, keeping(getattr(session, name)))
    return made


def untracked(session, made):
    """Frees the tensors `tracked` listed and gives the session its own calls back."""
    for name in ("tensor", "empty"):
        delattr(session, name)
    for tensor in made.values():
        tensor.free()


def compared(program, out, saved):
    """Each output against the saved one-island output: words that differ, and where
    any does, the relative RMS and largest error of the values."""
    result = {}
    for name, raw in out.items():
        key = f"{program}:{name}"
        if key not in saved:
            result[name] = "no one-island output"
            continue
        one = saved[key]
        unit = 4 if name in FLOAT32 else 2
        differ = int(np.sum(raw.view(f"u{unit}") != one.view(f"u{unit}")))
        entry = {"differ": differ, "words": int(raw.size // unit)}
        if differ:
            got, want = values(name, raw), values(name, one)
            scale = max(float(np.abs(want).max()), 1e-30)
            rms = float(np.sqrt(np.mean((got - want) ** 2) / np.mean(want**2)))
            entry |= {
                "rms_rel": rms,
                "max_rel": float(np.abs(got - want).max() / scale),
            }
        result[name] = entry
    return result


def main():
    args = parse()
    opened = dict(SHAPES[args.shape])
    if args.links:
        opened["links"] = args.links
    if args.kernel_bytes:
        opened["kernel_bytes"] = args.kernel_bytes
    saved = dict(np.load(args.against)) if args.against else {}
    report = {"shape": args.shape, "opened": opened, "programs": {}}
    keep = {}
    with km.Session(log=log, **opened) as session:
        graph = islands_of(session)
        if graph:
            report["islands"] = [
                [graph.first(i), graph.count(i)] for i in range(len(graph.islands))
            ]
        for program in args.programs.split(","):
            log(f"{program}: start")
            record = {}
            made = tracked(session)
            try:
                out, extra = RUNNERS[program](session, args)
                record |= extra
                record["status"] = "ran"
                record["words"] = compared(program, out, saved) if saved else {}
                keep |= {f"{program}:{name}": raw for name, raw in out.items()}
            except Exception as error:  # noqa: BLE001  # a refusal is a result
                record["status"] = "refused or failed"
                record["error"] = f"{type(error).__name__}: {error}"[:1500]
                record["where"] = traceback.format_exc().splitlines()[-6:]
            finally:
                untracked(session, made)
            log(f"{program}: {json.dumps(record)[:600]}")
            report["programs"][program] = record
    if args.save:
        np.savez(args.save, **keep)
    report["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(report)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
