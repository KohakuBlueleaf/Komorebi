"""One decode layer at T 256, the hand's (projects/decode) and the class's, from one x,
weights and cache: each output word for word against the hand's and float64, and a
layer's time each way. --chain: the class as one chain command (DecodeLayerChain,
overlays), its stamps' terms and the hand's chain beside it (README.md)."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from experiments.language.decode_class import decode_layer as dl
from komorebi import formats
from komorebi.bench import power
from komorebi.dsl import chain_stamps
from projects.decode import card
from projects.decode import model as md
from projects.dense import card as dense_card

STARTED = time.monotonic()
MODEL = dl.MODEL
STEPS = ("qkv", "attend", "o", "gate_up", "down")
# the layer's chain method's tensors, in its order
CHAIN_NAMES = ("x", "w_qkv", "qkv", "squares1", "qk", "keys", "values", "a", "table")
CHAIN_NAMES += ("w_o", "x1", "squares2", "w_gate_up", "gated", "w_down", "constants")
# the tensors each step writes, compared after it
WRITES = {
    "qkv": ("qkv",),
    "attend": ("a",),
    "o": ("x1", "squares2"),
    "gate_up": ("gated",),
    "down": ("x", "squares1"),
}


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10, help="timed layers a way")
    parser.add_argument("--steps", nargs="*", default=list(STEPS), choices=STEPS)
    parser.add_argument(
        "--chain", action="store_true", help="the overlay chain form (DecodeLayerChain)"
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="with --chain: o and down resident, three overlays (DecodeLayerMixed)",
    )
    parser.add_argument(
        "--kernel-bytes", type=int, default=None, help="the session's kernel buffer"
    )
    parser.add_argument(
        "--slots", default="default", help="--chain's slot counts a part, comma list"
    )
    return parser.parse_args()


def setup(session, seed):
    """The hand's program (a command a step, T 256) with layer 0's weights and cache
    written, and the host's copy of what it wrote."""
    decode = card.Decode(session, MODEL, 1, 0.5, log, context=dl.CONTEXT, chain=False)
    weights = md.layer_weights(MODEL, seed, 0)
    decode.write_layer(0, weights)
    cache = md.cache(MODEL, seed, 0, dl.CONTEXT)
    decode.write_cache(0, *cache)
    return decode, weights, cache


def class_tensors(session, decode):
    """The class's own activations (as the hand's, bf16 tiles) beside the hand's
    weights, cache, qk table and constants."""
    own = {
        name: session.empty(decode.tensors[name].shape, "bfloat16", "tile")
        for name in ("x", "x1", "qkv", "gated", "squares1", "squares2", "a")
    }
    w = decode.weights[0]
    return own | {
        "w_qkv": w["qkv"],
        "w_o": w["o"],
        "w_gate_up": w["gate_up"],
        "w_down": w["down"],
        "keys": w["keys"],
        "values": w["values"],
        "qk": decode.tensors["qk"],
        "constants": decode.tensors["constants"],
        "table": decode.tensors["attend"],
    }


def examples(t):
    """The class's example tensors by its handlers' parameter names."""
    return {name: t[name] for name in t if name not in ("squares1", "squares2")} | {
        "squares": t["squares1"],
        "attended": t["a"],
        "h": t["x1"],
        "g": t["gated"],
    }


def calls(program, t, position):
    """The layer's five commands: (step, handler, cores, words)."""
    plans = md.plans(MODEL, halves=(md.DEPTH, 1))
    cores = {name: plans[md.plan_of(MODEL, name)].cores() for name in dl.GROUPS}
    return [
        (
            "qkv",
            program.qkv,
            cores["qkv"],
            (t["x"], t["w_qkv"], t["qkv"], t["squares1"], t["qk"], "AB", position,
             0, 1),
        ),
        (
            "attend",
            program.attend,
            cores["o"],
            (t["qkv"], t["keys"], t["values"], t["a"], t["table"], position, 1),
        ),
        (
            "o",
            program.o,
            cores["o"],
            (t["a"], t["w_o"], t["x1"], t["x"], t["squares2"], t["constants"], "AB"),
        ),
        (
            "gate_up",
            program.gate_up,
            cores["gate_up"],
            (t["x1"], t["w_gate_up"], t["gated"], t["squares2"], t["constants"],
             "AAAB", 1),
        ),  # fmt: skip
        (
            "down",
            program.down,
            cores["down"],
            (t["gated"], t["w_down"], t["x"], t["x1"], t["squares1"], t["constants"],
             "AB"),
        ),  # fmt: skip
    ]


def start_class(session, t, start):
    """The class's x and its rows' squares (the cache is written before)."""
    session.write(t["x"], md.tile_pages(formats.bf16_bits(start)))
    squares = md.tile_pages(formats.bf16_bits(md.squares(start)))
    session.write(t["squares1"], squares)


def compared(session, t, names, hand, hand_values, want):
    """Each tensor's words against the hand's, its values against float64."""
    out = {}
    for name in names:
        same = bits(session, t[name]) == hand[name]
        out[f"{name}_words_differ"] = int(np.sum(~same))
        out[f"{name}_words"] = int(same.size)
        if name in hand_values:
            got = read(session, t[name])[:1]
            out[f"{name}_vs_float64"] = md.errors(got, want[name][:1])
    return out


def chained(session, decode, cache, start, t, hand, hand_values, want, args, count):
    """The layer as one chain command of DecodeLayerChain (--mixed: DecodeLayerMixed)
    with `count` slots a part (None: the default): once (the seven tensors against the
    hand's), timed, then stamped (each record's terms)."""
    position = dl.CONTEXT - 1
    names = [name for step in STEPS for name in WRITES[step]]
    out = {}
    for stamps in (False, True):
        decode.write_cache(0, *cache)
        start_class(session, t, start)
        form = dl.DecodeLayerMixed if args.mixed else dl.DecodeLayerChain
        program = form(session=session, log=log, chain_stamps=stamps, **examples(t))
        if count is not None:
            program.slots(count=count)
        run = session.load(program, wait=1.0, deadline=0.5)
        built = program.overlays_built.slots
        out["slots"] = {part: number for part, (number, _) in built.items()}
        try:
            chain = run.chain(
                program.layer(*(t[n] for n in CHAIN_NAMES), position), cores=(0, 120)
            )
            event = run.wait(run.submit(chain), timeout=5.0)
            if event.kind != "done":
                raise km.KomorebiError(f"chain: {event}")
            if not stamps:
                out["words"] = compared(session, t, names, hand, hand_values, want)
                out["first_us"] = event.microseconds
                print(json.dumps({f"chain {count}": out}), flush=True)  # before times
                times = [
                    run.wait(run.submit(chain), timeout=5.0).microseconds
                    for _ in range(args.runs)
                ]
                out["layer_us"] = round(float(np.median(times)), 2)
            else:
                rows_tensor = chain_stamps.tensor_for(session, 120)
                chain_stamps.read(session, run, program, rows_tensor, (0, 120))
                for _ in range(args.runs):
                    event = run.wait(run.submit(chain), timeout=5.0)
                    if event.kind != "done":
                        raise km.KomorebiError(f"stamped chain: {event}")
                rows = chain_stamps.read(session, run, program, rows_tensor, (0, 120))
                mhz = power.aiclk_mhz()
                out["terms"] = chain_stamps.terms(rows, mhz, len(STEPS))
                out["aiclk_mhz"] = mhz
        finally:
            run.stop()
        out["kernel" + ("_stamped" if stamps else "")] = run.result()["error_name"]
    return out


def hand_chain(session, weights, cache, args):
    """The hand's layer as its chain (card.Decode with chain=True, its own plans): a
    layer's time, args.runs tokens."""
    position = dl.CONTEXT - 1
    decode = card.Decode(session, MODEL, 1, 0.5, log, context=dl.CONTEXT, chain=True)
    decode.write_layer(0, weights)
    decode.write_cache(0, *cache)
    decode.write_activation("x", md.start(args.seed, MODEL))
    timed = decode.job(args.runs, 1, position=position - args.runs + 1, chain=True)
    decode.run.stop()
    slot = card.decode_slot(decode.stats()[timed["stats_slot"]])
    return {
        "layer_us": round(slot["microseconds"] / args.runs, 2),
        "kernel": decode.run.result()["error_name"],
    }


def chain_result(result):
    """--chain's last line: ok when every kernel ended clean and no word differs."""
    forms = result["chain"].values()
    kernels = [result["hand_kernel"], result["hand_chain"]["kernel"]]
    kernels += [out[k] for out in forms for k in ("kernel", "kernel_stamped")]
    result["ok"] = bool(
        all(k == "none" for k in kernels)
        and all(
            v == 0
            for out in forms
            for k, v in out["words"].items()
            if k.endswith("words_differ")
        )
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


def read(session, tensor):
    return dense_card.read_rows(session, tensor, range(md.TILE))


def bits(session, tensor):
    return session.read(tensor)[0].view(np.uint16)


def main():
    args = parse()
    position = dl.CONTEXT - 1
    start = md.start(args.seed, MODEL)
    result = {"context": dl.CONTEXT, "position": position, "steps": {}}
    # the class's five handlers resident (a chain's walk takes no overlays): 170,528
    # B of kernel text, which the session's kernel buffer follows
    with km.Session(log=log, kernel_bytes=args.kernel_bytes) as session:
        decode, weights, cache = setup(session, args.seed)
        cached = tuple(c[:, :position].copy() for c in cache)
        want = md.layer(MODEL, start, weights, cached, 1)
        want["qkv"] = np.concatenate([want["q"], want["k"], want["v"]], axis=1)
        decode.write_activation("x", start)
        decode.job(1, 1, position=position, chain=False)
        hand = {
            name: bits(session, decode.tensors[name])
            for names in WRITES.values()
            for name in names
        }
        hand_values = {
            name: read(session, decode.tensors[name])
            for name in ("qkv", "a", "x1", "gated", "x")
        }
        timed = decode.job(
            args.runs, 1, position=position - args.runs + 1, chain=False
        )  # a token a layer, args.runs of them
        decode.run.stop()
        result["hand_kernel"] = decode.run.result()["error_name"]
        slot = card.decode_slot(decode.stats()[timed["stats_slot"]])
        result["hand_layer_us"] = round(slot["microseconds"] / args.runs, 2)
        result["hand_steps_us"] = {
            name: round(slot["steps"][name]["sum_us"] / args.runs, 2) for name in STEPS
        }
        decode.write_cache(0, *cache)  # the hand's runs appended the token: as before
        t = class_tensors(session, decode)
        if args.chain:
            result["chain"] = {
                count: chained(
                    session,
                    decode,
                    cache,
                    start,
                    t,
                    hand,
                    hand_values,
                    want,
                    args,
                    None if count == "default" else int(count),
                )
                for count in args.slots.split(",")
            }
            result["hand_chain"] = hand_chain(session, weights, cache, args)
            return chain_result(result)
        start_class(session, t, start)
        program = dl.DecodeLayer(session=session, log=log, **examples(t))
        run = session.load(program, wait=1.0, deadline=0.5)
        layer = calls(program, t, position)
        for step, handler, cores, words in layer:
            if step not in args.steps:
                continue
            handle = run.submit(handler, (0, cores), args=words, exclusive=True)
            event = run.wait(handle, timeout=5.0)
            if event.kind != "done":
                raise km.KomorebiError(f"{step}: {event}")
            out = {"us": event.microseconds}
            for name in WRITES[step]:
                got = bits(session, t[name])
                same = got == hand[name]
                out[f"{name}_words_differ"] = int(np.sum(~same))
                out[f"{name}_words"] = int(same.size)
            for name in WRITES[step]:
                if name in hand_values:
                    got = read(session, t[name])[:1]
                    out[f"{name}_vs_float64"] = md.errors(got, want[name][:1])
                    out[f"{name}_hand_vs_float64"] = md.errors(
                        hand_values[name][:1], want[name][:1]
                    )
            result["steps"][step] = out
        print(json.dumps({"words": result["steps"]}), flush=True)  # before the times
        if args.runs and set(args.steps) == set(STEPS):
            times = []
            for _ in range(args.runs):
                handles = [
                    run.submit(handler, (0, cores), args=words, exclusive=True)
                    for _, handler, cores, words in layer
                ]
                events = run.wait(handles, timeout=10.0)
                if {e.kind for e in events} != {"done"}:
                    raise km.KomorebiError(f"layer: {[e.kind for e in events]}")
                times.append([e.microseconds for e in events])
            times = np.array(times)
            result["class_steps_us"] = {
                name: round(float(np.median(times[:, i])), 2)
                for i, name in enumerate(STEPS)
            }
            result["class_layer_us"] = round(float(np.median(times.sum(axis=1))), 2)
        run.stop()
        result["class_kernel"] = run.result()["error_name"]
    result["ok"] = bool(
        result["hand_kernel"] == "none"
        and result["class_kernel"] == "none"
        and all(
            v == 0
            for step in result["steps"].values()
            for k, v in step.items()
            if k.endswith("words_differ")
        )
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
