"""Staged run of the softmax step's rules on one core (README.md, "The rules, one at
a time"): the hand step, the emitted step with no rule, one rule added a stage, BODY
20 and 21, then the products 22 and 23; each stage's name is logged before its launch.
`rule_stages.py build` (mock device) and `rule_stages.py card` (compiles nothing)."""

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import rules  # noqa: E402
from rules import BODIES, GENERATED, NONCE, TILE_WORDS, UNITS, log  # noqa: E402

import komorebi as km  # noqa: E402
from komorebi.device import native  # noqa: E402
from komorebi.dsl.tile import emit, steps  # noqa: E402
from komorebi.session import kernels as session_kernels  # noqa: E402
from komorebi.session import process  # noqa: E402

STAGED = GENERATED / "stages"
# The step's rules: 1 matrix_sum, 2 max_macro, 3 interleave, 4 LReg 11 a constant
# register, 5 constants held for the step (the exp step's: its store's address mode).
ALL = frozenset({1, 2, 3, 4, 5})
# (name, BODY, rules on; None: rules.py's own header).
STAGES = [
    ("s0 hand step (run on the card before)", 14, None),
    ("s1 emitted step, no rule", 15, frozenset()),
    ("s2 + constants held once a step (rule 5)", 15, frozenset({5})),
    ("s3 + LReg 11 a constant register (rule 4)", 15, frozenset({4, 5})),
    ("s4 + interleave (rule 3)", 15, frozenset({3, 4, 5})),
    ("s5 + max macro (rule 2)", 15, frozenset({2, 3, 4, 5})),
    ("s6 + matrix row sums (rule 1): body 20", 20, None),
    ("s7 exp step, its rules: body 21", 21, None),
    ("p8 product, the library's matmul_block_no_mop: body 22", 22, None),
    ("p9 product, the emitted MVMUL words: body 23", 23, None),
]


@contextlib.contextmanager
def rules_on(on):
    """komorebi.dsl.tile.steps with only the rules `on` (the others' entry points
    replaced while a stage's header is built)."""
    saved = {
        k: getattr(steps, k)
        for k in ("_matrix_sums", "max_macro", "interleave", "constant_registers")
    }
    attempt = steps._attempt
    if 1 not in on:
        steps._matrix_sums = lambda *a: (set(), ["off"])
    if 2 not in on:
        steps.max_macro = lambda code: ([], ["off"])
    if 3 not in on:
        steps.interleave = lambda code, start, end: 0
    if 4 not in on:
        count = len(steps.CONSTANT_REGISTERS) - 1

        def constant_registers(code, prog, registers=None):
            return saved["constant_registers"](code, prog, count)

        steps.constant_registers = constant_registers
    if 5 not in on:

        def no_whole(name, dest, plan, whole, cacheable):
            return attempt(name, dest, plan, set(), cacheable | whole)

        steps._attempt = no_whole
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(steps, k, v)
        steps._attempt = attempt


def stage_folder(k):
    return STAGED / f"s{k}"


def generate():
    """The stages' headers (softstep.hpp a folder) and their estimates."""
    costs = {}
    for k, (name, _, on) in enumerate(STAGES):
        if on is None:
            continue
        with rules_on(on):
            unit = steps.step_program(
                rules.softmax_ir(), rules.STEP_WIDTH, rules.STEP_TILE
            )
        stage_folder(k).mkdir(parents=True, exist_ok=True)
        (stage_folder(k) / "softstep.hpp").write_text(emit(unit))
        costs[name] = unit.cost("math")[1]
    log(f"stage estimates {json.dumps(costs)}")
    saved = {
        "sources": rules.sources_hash() + Path(__file__).read_text(),
        "costs": costs,
    }
    (STAGED / "costs.json").write_text(json.dumps(saved))
    return costs


def saved_costs():
    """The estimates `build` wrote; refuses headers built from other sources."""
    try:
        saved = json.loads((STAGED / "costs.json").read_text())
    except (OSError, ValueError):
        sys.exit("no stage headers: run `rule_stages.py build` first, outside the job")
    if saved["sources"] != rules.sources_hash() + Path(__file__).read_text():
        sys.exit("the stage headers are stale: run `rule_stages.py build` again")
    return saved["costs"]


def launch(session, k, data):
    _, body, on = STAGES[k]
    stage = None if on is None else (k, stage_folder(k))
    return rules.launch(session, body, data, stage)


def case_of(k):
    return BODIES[STAGES[k][1]][0]


def evaluate(k, data, out, info, nonce):
    """The stage's record (the experiment's check of its body)."""
    body = STAGES[k][1]
    rec = rules.check(case_of(k), data, {body: out}, {body: info}, {body: nonce})
    got = rec[BODIES[body][1]]
    return {"stage": STAGES[k][0], **got}


def build(args):
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / rules.DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(rules.JIT_CACHE)
    rules.generate()
    generate()
    with km.Session() as session:
        for k, (name, *_) in enumerate(STAGES):
            data = rules.inputs_of(case_of(k), args.units, args.seed)
            run, *_ = launch(session, k, data)
            _, *built = run.build()
            got = session_kernels.kernel_sizes(built)
            log(f"built {name}: kernel buffer {got.total} of {got.limit} B")
    for k, (_, body, _) in enumerate(STAGES):
        case, form = BODIES[body]
        data = rules.inputs_of(case, args.units, args.seed)
        if case == "softmax":
            out = rules.softmax_words(data, form)
        else:
            out = rules.reference(case, data).astype(np.float32).view(np.uint32)
        info = np.zeros(TILE_WORDS, np.uint32)
        info[NONCE], info[UNITS] = 7, args.units
        log(json.dumps(evaluate(k, data, out, info, 7)))
    return 0


def card(args):
    base = rules.generated_costs()
    costs = saved_costs()
    costs[STAGES[6][0]] = base["softstep"]
    costs[STAGES[7][0]] = base["expstep_flash"]
    costs[STAGES[9][0]] = base[f"product_{rules.PRODUCT_FIDELITY.lower()}"]
    os.environ["TT_METAL_CACHE"] = str(rules.JIT_CACHE)
    log(f"kernels from the build's cache {rules.JIT_CACHE}")
    started, records, saved, products = time.time(), [], {}, {}
    with km.Session() as session:
        log(f"device open {time.time() - started:.1f}s")
        for k, (name, body, _) in enumerate(STAGES):
            if args.stages is not None and k not in args.stages:
                continue
            data = rules.inputs_of(case_of(k), args.units, args.seed)
            run, t_out, t_info, nonce = launch(session, k, data)
            log(f"launch {name}")
            run.run()
            session.device.finish()
            log(f"returned {name}")
            out = t_out.read().reshape(-1, TILE_WORDS).copy()
            info = t_info.read().reshape(-1).copy()
            rec = evaluate(k, data, out, info, nonce)
            rec["estimate"] = costs.get(name)
            log(json.dumps(rec))
            records.append(rec)
            saved[f"s{k}"], saved[f"s{k}_info"] = out, info
            if case_of(k) == "product":
                products[body] = (data, out, info, nonce)
    if len(products) == 2:
        # the emitted words against the library's call: values and cycles
        data = products[22][0]
        outs, infos, nonces = (
            {b: v[i] for b, v in products.items()} for i in (1, 2, 3)
        )
        log(f"products {json.dumps(rules.check('product', data, outs, infos, nonces))}")
    if args.save:
        np.savez(args.save, **saved)
    print(json.dumps({"wall_s": round(time.time() - started, 1)}), flush=True)
    return 0 if all(r["nonce_ok"] and r["error"] == 0 for r in records) else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["build", "card"])
    p.add_argument("--units", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", help="card: the stages' outputs into this .npz file")
    p.add_argument("--stages", nargs="+", type=int, help="card: only these stages")
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
