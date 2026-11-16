"""The decode layer's class chain built once with its position given at submit
(km.at_submit("position")), submitted for positions 0, 1 and 2, each output word for
word against the hand's command-a-step layer at that position and against float64.
Example: python at_submit.py --positions 0 1 2 --dump out.npz (README.md)"""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from experiments.language.decode_class import decode_layer as dl
from experiments.language.decode_class import run_layer as rl
from komorebi import prebuild
from projects.decode import model as md

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def where(got, want):
    """Where two tensors' words differ: the count, the first 16 word indices, and
    the largest difference of their bf16 values."""
    wrong = np.flatnonzero(got != want)
    values = [
        (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        for w in (got, want)
    ]
    return {
        "differ": int(wrong.size),
        "first": wrong[:16].tolist(),
        "largest": float(np.abs(values[0] - values[1]).max()),
    }


def hand_outputs(session, decode, cache, start, positions):
    """The hand's layer (a command a step) at each position, the cache as written
    before each: {position: {tensor: its bf16 words}}."""
    names = [name for step in rl.STEPS for name in rl.WRITES[step]]
    out = {}
    for position in positions:
        if prebuild.active():  # the hand's job reads the X280 thread's slots: built
            out[position] = {n: rl.bits(session, decode.tensors[n]) for n in names}
            continue
        decode.write_cache(0, *cache)
        decode.write_activation("x", start)
        decode.job(1, 1, position=position, chain=False)
        out[position] = {n: rl.bits(session, decode.tensors[n]).copy() for n in names}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--kernel-bytes", type=int, default=102400)
    parser.add_argument("--dump", help="every position's outputs, both ways, .npz")
    args = parser.parse_args()
    start = md.start(args.seed, MODEL)
    names = [name for step in rl.STEPS for name in rl.WRITES[step]]
    result = {"positions": args.positions, "steps": {}}
    dumped = {}
    with km.Session(log=log, kernel_bytes=args.kernel_bytes) as session:
        decode, weights, cache = rl.setup(session, args.seed)
        hand = hand_outputs(session, decode, cache, start, args.positions)
        decode.run.stop()
        result["hand_kernel"] = decode.run.result()["error_name"]
        t = rl.class_tensors(session, decode)
        program = dl.DecodeLayerChain(session=session, log=log, **rl.examples(t))
        run = session.load(program, wait=1.0, deadline=0.5)
        try:
            chain = run.chain(  # built once: the records do not hold the position
                program.layer(
                    *(t[n] for n in rl.CHAIN_NAMES), km.at_submit("position")
                ),
                cores=(0, 120),
            )
            result["chain_word"] = chain.word
            for position in args.positions:
                decode.write_cache(0, *cache)
                rl.start_class(session, t, start)
                event = run.wait(run.submit(chain, position=position), timeout=5.0)
                cached = tuple(c[:, :position].copy() for c in cache)
                want = md.layer(MODEL, start, weights, cached, 1)
                out = {"event": event.kind, "us": event.microseconds}
                for name in names:
                    got = rl.bits(session, t[name]).copy()
                    out[name] = where(got, hand[position][name])
                    dumped[f"class_{position}_{name}"] = got
                    dumped[f"hand_{position}_{name}"] = hand[position][name]
                out["x_vs_float64"] = md.errors(
                    rl.read(session, t["x"])[:1], want["x"][:1]
                )
                result["steps"][position] = out
        finally:
            run.stop()
        result["class_kernel"] = run.result()["error_name"]
    if args.dump:
        np.savez_compressed(args.dump, **dumped)
    result["ok"] = result["hand_kernel"] == result["class_kernel"] == "none" and all(
        out["event"] == "done" and all(out[n]["differ"] == 0 for n in names)
        for out in result["steps"].values()
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


MODEL = dl.MODEL

if __name__ == "__main__":
    sys.exit(main())
