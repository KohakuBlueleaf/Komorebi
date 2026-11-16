"""The decode class's layer as one chain whose marks cross islands (on two cards each
step's groups inside an island, the next step's cores on both). Its seven tensors saved
(--save) or compared word for word with a saved run (--against); --steps: a command a
step beside it. One JSON line."""

import argparse
import json
import sys
import time

import numpy as np

import komorebi as km
from experiments.language.decode_class import decode_layer as dl
from experiments.language.decode_class import run_layer as rl
from experiments.language.decode_class.decode_layer import (
    DOWN_CORES,
    GATE_UP_CORES,
    O_CORES,
    QKV_CORES,
)
from komorebi import chain as kc
from komorebi import power
from projects.decode import model as md

STARTED = time.monotonic()
MODEL = dl.MODEL
NAMES = [name for step in rl.STEPS for name in rl.WRITES[step]]


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cards", default="0", help="one card, or two: 0,1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10, help="timed layers")
    parser.add_argument(
        "--firsts",
        default=None,
        help="first ranks of qkv, attend/o/down, gate_up (default: 0,0,0 on one "
        "card; 6,103,70 on two: island 1 begins at 118)",
    )
    parser.add_argument(
        "--steps", action="store_true", help="also a command a step at the same ranks"
    )
    parser.add_argument(
        "--prefixes",
        action="store_true",
        help="also the chains of the first 1 .. 5 calls timed, each record's far marks",
    )
    parser.add_argument("--steps-only", action="store_true", help="no chain: steps")
    parser.add_argument("--save", default=None, help="the tensors' words to this .npz")
    parser.add_argument("--against", default=None, help="compare with this .npz")
    return parser.parse_args()


class DecodeLayerAcross(dl.DecodeLayerChain):
    """DecodeLayerChain's five calls, each step from its own first rank."""

    @km.chain
    def placed(
        self,
        x,
        w_qkv,
        qkv,
        squares1,
        qk,
        keys,
        values,
        a,
        table,
        w_o,
        x1,
        squares2,
        w_gate_up,
        gated,
        w_down,
        constants,
        position: int,
        qkv_first: int,
        o_first: int,
        up_first: int,
    ):
        """The layer's calls as DecodeLayerChain.layer's, from the given ranks."""
        self.qkv(
            x,
            w_qkv,
            qkv,
            squares1,
            qk,
            "AB",
            position,
            0,
            1,
            cores=(qkv_first, QKV_CORES),
        )
        self.attend(qkv, keys, values, a, table, position, 1, cores=(o_first, O_CORES))
        self.o(a, w_o, x1, x, squares2, constants, "AB", cores=(o_first, O_CORES))
        self.gate_up(
            x1,
            w_gate_up,
            gated,
            squares2,
            constants,
            "AAAB",
            1,
            cores=(up_first, GATE_UP_CORES),
        )
        self.down(
            gated,
            w_down,
            x,
            km.past(x1),
            squares1,
            constants,
            "AB",
            cores=(o_first, DOWN_CORES),
        )


def tensors_of(session, weights, cache, two):
    """The class's tensors: the weights, cache and tables on every island (each reads
    its own), the activations split as the session places them."""
    hint = {"copy": True} if two else {}
    products = MODEL.products()
    widths = {"x": MODEL.hidden, "x1": MODEL.hidden, "gated": MODEL.mlp}
    widths |= dict.fromkeys(("squares1", "squares2"), md.D["DC_PARTS"] * md.TILE)
    widths |= {"qkv": products["qkv"][1], "a": MODEL.heads * MODEL.head_dim}
    t = {
        name: session.empty((md.TILE, widths[name]), "bfloat16", "tile")
        for name in widths
    }
    shapes = products | {name: MODEL.shapes()[name] for name in md.APART}
    rows = (MODEL.kv_heads * dl.CONTEXT, MODEL.head_dim)
    shapes |= dict.fromkeys(("keys", "values"), rows)
    held = {
        name: session.empty(shape, "bfloat16", "tile", **hint)
        for name, shape in shapes.items()
    }
    for name, pages in md.fused(weights).items():
        session.write(held[name], pages)
    session.write(held["keys"], md.cache_pages(cache[0], dl.CONTEXT))
    session.write(held["values"], md.cache_pages(cache[1], dl.CONTEXT))
    qk = md.qk_pages(MODEL, dl.CONTEXT + md.TILE)  # layer 0's head norms in place
    pages = md.qk_weight_pages(weights["q_norm"][1], weights["k_norm"][1])
    qk[md.D["DC_QK_WEIGHTS"] : md.D["DC_QK_WEIGHTS"] + len(pages)] = pages
    t["qk"] = session.empty((md.TILE, md.TILE * len(qk)), "bfloat16", "tile", **hint)
    session.write(t["qk"], qk)
    for name, values in (
        ("constants", md.constants()),
        ("table", md.attend_constants()),
    ):
        values = np.asarray(values, np.float32)
        t[name] = session.tensor(values, "bfloat16", "tile", **hint)
    return t | {
        "w_qkv": held["qkv"],
        "w_o": held["o"],
        "w_gate_up": held["gate_up"],
        "w_down": held["down"],
        "keys": held["keys"],
        "values": held["values"],
    }


def run_zero(run):
    """The run that describes a class's calls: island 0's on islands."""
    return run.runs[0] if hasattr(run, "runs") else run


def far_marks(calls, edges):
    """Each record's marks (root, core) as the walk issues them (chain.prove's
    relation), and how many go to another island (`edges`: each island's first rank)."""

    def island(rank):
        return sum(rank >= edge for edge in edges)

    out = []
    for index in range(1, len(calls)):
        call, before = calls[index], calls[index - 1]
        groups = before.group[1] // before.size
        pairs = {
            (before.group[0] + block * before.size, core)
            for core, rank, group in kc._cores(call)
            for first, count in kc.tiles_of(call.k, call.ranges, rank, group, call.size)
            for block in kc.blocks_met(first, count, before.width, groups)
        }
        far = [p for p in pairs if island(p[0]) != island(p[1])]
        roots = {}
        for root, _ in far:
            roots[root] = roots.get(root, 0) + 1
        out.append(
            {
                "record": index,
                "marks": len(pairs),
                "far": len(far),
                "far_a_root_most": max(roots.values(), default=0),
            }
        )
    return out


def by_steps(run, steps):
    """The layer as a command a step, each waited for: its microseconds summed."""
    total = 0.0
    for handler, cores, words in steps:
        event = run.wait(run.submit(handler, cores, args=words, exclusive=True))
        if event.kind != "done":
            raise km.KomorebiError(f"step: {event}")
        total += event.microseconds
    return total


def main():
    args = parse()
    cards = [int(c) for c in args.cards.split(",")]
    two = len(cards) > 1
    firsts = args.firsts or ("6,103,70" if two else "0,0,0")
    qkv_first, o_first, up_first = (int(f) for f in firsts.split(","))
    position = dl.CONTEXT - 1
    start = md.start(args.seed, MODEL)
    weights = md.layer_weights(MODEL, args.seed, 0)
    cache = md.cache(MODEL, args.seed, 0, dl.CONTEXT)
    result = {"cards": cards, "firsts": [qkv_first, o_first, up_first]}
    with km.Session(log=log, **({"cards": cards} if two else {})) as session:
        t = tensors_of(session, weights, cache, two)
        rl.start_class(session, t, start)
        program = DecodeLayerAcross(session=session, log=log, **rl.examples(t))
        run = session.load(program, wait=1.0, deadline=0.5)
        ends = [
            (qkv_first, QKV_CORES),
            (o_first, O_CORES),
            (up_first, GATE_UP_CORES),
        ]
        low = min(f for f, _ in ends)
        high = max(f + n for f, n in ends)
        words = [t[n] for n in rl.CHAIN_NAMES]
        chain = None
        if args.steps_only:  # the five steps as commands, no chain
            first = {"qkv": qkv_first, "gate_up": up_first}
            steps = [
                (handler, (first.get(step, o_first), cores), step_words)
                for step, handler, cores, step_words in rl.calls(program, t, position)
            ]
            layer_times = [by_steps(run, steps)]
            got = {name: rl.bits(session, t[name]) for name in NAMES}
            layer_times += [by_steps(run, steps) for _ in range(args.runs)]
            result["steps_layer_us"] = round(float(np.median(layer_times[1:])), 2)
        else:
            chain = run.chain(
                program.placed(*words, position, qkv_first, o_first, up_first),
                cores=(low, high - low),
            )
            event = run.wait(run.submit(chain), timeout=10.0)
            if event.kind != "done":
                raise km.KomorebiError(f"chain: {event}")
            got = {name: rl.bits(session, t[name]) for name in NAMES}
            result["first_us"] = event.microseconds
            times = [
                run.wait(run.submit(chain), timeout=10.0).microseconds
                for _ in range(args.runs)
            ]
            result["layer_us"] = round(float(np.median(times)), 2)
        if args.prefixes and chain is not None:  # the layer's time a record at a time
            layer = program.placed(*words, position, qkv_first, o_first, up_first)
            group = (low, high - low)
            edges = (
                [session.graph.first(i) for i in range(1, len(cards))] if two else []
            )
            try:  # the count is the host's: an error in it is printed, not fatal
                calls = run_zero(run)._class_calls(layer, None, True, group)
                result["records"] = far_marks(calls, edges)
            except Exception as error:  # noqa: BLE001
                result["records"] = f"not counted: {type(error).__name__}: {error}"
            print(json.dumps({"records": result["records"]}), flush=True)
            prefix_us = []
            for count in range(1, len(layer) + 1):
                part = run.chain(layer[:count], cores=group)
                spans = [
                    run.wait(run.submit(part), timeout=10.0).microseconds
                    for _ in range(args.runs)
                ]
                part.free()
                prefix_us.append(round(float(np.median(spans)), 2))
            result["prefix_us"] = prefix_us
        # a command a step at the same ranks (no marks): its words, then its times
        if args.steps and chain is not None:
            rl.start_class(session, t, start)
            first = {"qkv": qkv_first, "gate_up": up_first}
            steps = [
                (handler, (first.get(step, o_first), cores), step_words)
                for step, handler, cores, step_words in rl.calls(program, t, position)
            ]
            layer_times = [by_steps(run, steps)]
            stepped = {name: rl.bits(session, t[name]) for name in NAMES}
            result["steps_words_differ"] = {
                name: int(np.sum(stepped[name] != got[name])) for name in NAMES
            }
            layer_times += [by_steps(run, steps) for _ in range(args.runs)]
            result["steps_layer_us"] = round(float(np.median(layer_times[1:])), 2)
        run.stop()
        if chain is not None:
            chain.free()
        blocks = run.result()  # a list over islands
        blocks = blocks if isinstance(blocks, list) else [blocks]
        result["kernel"] = [block["error_name"] for block in blocks]
    if args.save:
        np.savez(args.save, **got)
    differ = None
    if args.against:
        saved = np.load(args.against)
        differ = {name: int(np.sum(got[name] != saved[name])) for name in NAMES}
        result["words_differ"] = differ
        result["words"] = {name: int(got[name].size) for name in NAMES}
    result["ok"] = bool(
        all(k == "none" for k in result["kernel"])
        and not any(result.get("steps_words_differ", {}).values())
        and (differ is None or not any(differ.values()))
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
