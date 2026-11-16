"""One decode layer on the card at T 256: the hand's (projects/decode, a command a
step) and the class's (decode_layer.py, its five handlers a command each) from the
same x, weights and cache; each output word for word against the hand's and against
float64 (model.layer), and the time of a layer each way."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from experiments.language.decode_class import decode_layer as dl
from komorebi import formats
from projects.decode import card
from projects.decode import model as md
from projects.dense import card as dense_card

STARTED = time.monotonic()
MODEL = dl.MODEL
STEPS = ("qkv", "attend", "o", "gate_up", "down")
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
    with km.Session(log=log) as session:
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
        session.write(t["x"], md.tile_pages(formats.bf16_bits(start)))
        squares = md.tile_pages(formats.bf16_bits(md.squares(start)))
        session.write(t["squares1"], squares)
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
