"""Staged run of the per-core attention body, one session, one core: each stage's
name is printed before its launch and "returned" after it.

`attn_stages.py build` (mock device) and `attn_stages.py card` (stops at a stall).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CACHE = HERE.parents[2] / ".logs" / "jit-rules" / "attn-stages"
os.environ["TT_METAL_CACHE"] = str(CACHE)

import numpy as np  # noqa: E402
import rules  # noqa: E402
from rules import log, matrix_of, tiles_of  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi import native  # noqa: E402
from komorebi import session as km_session  # noqa: E402

# (name, BODY, STAGE, steps): a the score products, b plus the hand step (both packed
# out), c the whole body 18 (1, then 16 steps), d body 19 (16 steps).
STAGES = [
    ("a reader + score product, packed out", 18, 1, 1),
    ("b + hand softmax step, packed out", 18, 2, 1),
    ("c1 whole body 18, 1 step", 18, 3, 1),
    ("c2 whole body 18, 16 steps", 18, 3, 16),
    ("d body 19 (emitted step), 16 steps", 19, 3, 16),
]


def evaluate(name, body, stage, data, out, info, nonce):
    """The stage's verdict from what came back."""
    rec = {"stage": name, "nonce_ok": int(info[rules.NONCE]) == nonce}
    rec["error"] = int(info[rules.ERROR])
    words = out.view(np.float32)
    rec["finite"] = bool(np.isfinite(words).all())
    if stage == 1:
        # Out tiles 0..2: the raw scores Q K^T of the step's 3 key tiles.
        got = matrix_of(words[:3]).astype(np.float64)
        ref = data["q64"] @ data["k64"][:96].T
        rec["score_error"] = float(np.abs(got - ref).max() / np.abs(ref).max())
        rec["ok"] = rec["nonce_ok"] and rec["finite"] and rec["score_error"] < 4e-3
    elif stage == 2:
        p = words[: 3 * rules.TILE_WORDS]
        rec["p_range"] = [float(p.min()), float(p.max())]
        rec["ok"] = rec["nonce_ok"] and rec["finite"]
    else:
        full = rules.check_attn(data, {body: out}, {body: info}, {body: nonce})
        rec["record"] = full[rules.BODIES[body][1]]
        rec["ok"] = bool(full[rules.BODIES[body][1]]["ok"])
    return rec


def made_up(stage, body, data, steps, nonce):
    """A result of the right shape (build mode's check of the host side)."""
    info = np.zeros(rules.TILE_WORDS, np.uint32)
    info[rules.NONCE], info[rules.UNITS] = nonce, steps
    info[rules.SYNC : rules.SYNC + rules.SYNCS] = 10
    info[rules.BODY : rules.BODY + steps] = 100
    info[rules.LOOP : rules.LOOP + steps] = 200
    for at in (64, 256, 320):
        info[at : at + steps] = 50
    if stage == 1:
        ref = (data["q64"] @ data["k64"][:96].T).astype(np.float32)
        tiles = list(tiles_of(ref)) + [np.zeros(rules.TILE_WORDS, np.float32)]
        out = np.stack(tiles).astype(np.float32).view(np.uint32)
    elif stage == 2:
        out = np.ones((4, rules.TILE_WORDS), np.float32).view(np.uint32)
    else:
        out = rules.attn_words(data, rules.BODIES[body][1])
    return out, info


def build(args):
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / rules.DESCRIPTOR)
    km_session.chip_holder = lambda card, path=None: None  # a mock device has no chip
    rules.generate()
    sizes = {}
    with km.Session() as session:
        for name, body, stage, steps in STAGES:
            data = rules.attn_inputs(steps, args.seed)
            log(f"build {name}")
            run, *_ = rules.launch_attn(session, body, data, stage)
            _, *built = run.build()
            got = km_session.kernel_sizes(built)
            sizes[name] = [got.total, got.limit, got.text]
            log(f"built {name}: kernel buffer {got.total} of {got.limit} B")
    for name, body, stage, steps in STAGES:
        data = rules.attn_inputs(steps, args.seed)
        out, info = made_up(stage, body, data, steps, 7)
        log(json.dumps(evaluate(name, body, stage, data, out, info, 7)))
    print(json.dumps({"kernel_buffer": sizes}), flush=True)
    return 0


def card(args):
    started = time.time()
    done = []
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for name, body, stage, steps in STAGES:
            if args.only and name.split()[0] not in args.only:
                continue
            data = rules.attn_inputs(steps, args.seed)
            log(f"stage {name}: launching")
            run, t_out, t_info, nonce = rules.launch_attn(session, body, data, stage)
            try:
                run.run()
                session.device.finish()
            except Exception as error:
                log(f"stage {name}: FAILED {type(error).__name__}: {str(error)[:200]}")
                raise
            log(f"stage {name}: returned")
            out = t_out.read().reshape(-1, rules.TILE_WORDS).copy()
            info = t_info.read().reshape(-1).copy()
            rec = evaluate(name, body, stage, data, out, info, nonce)
            log(json.dumps(rec))
            done.append(rec)
            if args.save:
                np.savez(f"{args.save}-{name.split()[0]}.npz", out=out, info=info)
    print(json.dumps({"wall_s": round(time.time() - started, 1)}), flush=True)
    return 0 if all(r["ok"] for r in done) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", nargs="+", help="card: only these stages (a b c1 c2 d)")
    p.add_argument("--save", help="card: outputs into <save>-<stage>.npz")
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
