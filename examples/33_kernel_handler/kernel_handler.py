"""A hand-written kernel as a handler of a class: `gate` is example 22's kernels as
functions (km.kernel), beside `add` in the DSL, in one class or in two merged, and
(--hand) the same functions as a hand program of the class's compute options. Every
command's output is checked. One JSON line. Example: python kernel_handler.py --hand
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi.bench import power

HERE = Path(__file__).resolve().parent
HAND = HERE / "kernels" / "hand"  # the same functions as a program of their own
sys.path.insert(0, str(HERE.parent / "22_gate"))

import gate as hand  # noqa: E402  (the hand program, its values and its check)

ALTERNATIONS = 20  # gate then add, each checked
# The gate's hand kernels: one function a role, gate_k::gate_movement on both movers
# and gate_k::gate_compute on the three compute CPUs.
GATE = km.kernel(
    movement="kernels/gate_movement.hpp",
    compute="kernels/gate_compute.hpp",
    entry="gate_k::gate",
)


@km.tile_function
def add_tiles(a, b):
    """The sum of two tiles: what the tile compiler emits for compute."""
    return a + b


class Mixed(km.Program):
    """gate by hand kernels through three declared pairs; add in the DSL, its own
    streams. Each core takes the pages from rank x tiles for tiles pages."""

    BLOCK = km.constant(4)  # tiles a block: one acquire of a float32 Dst's half
    a_in = km.Pair(km.bf16, pages=BLOCK * 4)  # four blocks a half
    b_in = km.Pair(km.bf16, pages=BLOCK * 4)
    result = km.Pair(km.bf16, pages=BLOCK * 4, write=True)

    @km.handler(GATE)
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        """out = silu(a) * b."""

    @km.handler
    def add(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(add_tiles(x, y))


class GateOnly(km.Program):
    """Mixed's gate alone, for --merged."""

    BLOCK = km.constant(4)
    a_in = km.Pair(km.bf16, pages=BLOCK * 4)
    b_in = km.Pair(km.bf16, pages=BLOCK * 4)
    result = km.Pair(km.bf16, pages=BLOCK * 4, write=True)

    @km.handler(GATE)
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        """out = silu(a) * b."""


class AddOnly(km.Program):
    """Mixed's add alone: for --merged, and for --add-alone's time."""

    BLOCK = km.constant(4)

    @km.handler
    def add(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for x, y in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(add_tiles(x, y))


def judged(session, out, wanted):
    """The output against float64 by the hand program's check: values outside its
    bound (relative, plus 2^-10), and the largest relative error over values of 2^-4
    or more."""
    wrong, relative = 0, 0.0
    for first, got in session.read_chunks(out):
        got = hand.floats(got.view(np.uint16).reshape(len(got), -1))
        want = wanted[first : first + len(got)]
        difference = np.abs(got - want)
        large = np.abs(want) >= hand.SMALL
        relative = max(relative, float((difference / np.abs(want))[large].max()))
        wrong += int((difference > hand.BOUND * np.abs(want) + 2.0**-10).sum())
    return wrong, relative


class Bench:
    """A loaded program's handlers on `cores` cores: each command's output cleared
    before it and judged after it, its time on its event; `words` the hand program's
    output, which each gate's is compared with word for word."""

    def __init__(self, session, run, tensors, args, cores, words=None):
        self.session, self.run, self.args, self.cores = session, run, args, cores
        self.a, self.b, self.outs, self.wanted = tensors
        self.pages, self.words = cores * args.tiles, words

    def call(self, name, handler):
        """One command of `handler` (None: the hand program's GATE) into its output:
        (done and 0 wrong, wrong, the largest relative error, its microseconds, its
        words equal to the hand program's, or None)."""
        out = self.outs[name]
        self.session.write(out, np.zeros((self.pages, 2048), np.uint8))
        if handler is None:  # the hand program: its handler's name and raw words
            words = [self.a.address, self.b.address, out.address]
            words += [self.args.tiles, self.args.pattern.word]
            handle = self.run.submit("gate", (0, self.cores), words)
        else:
            words = (self.a, self.b, out, self.args.tiles, self.args.pattern)
            handle = self.run.submit(handler, (0, self.cores), args=words)
        event = self.run.wait(handle, timeout=30.0)
        wrong, relative = judged(self.session, out, self.wanted[name])
        if event.kind != "done":
            hand.log(f"{name}: {event}")
        same = None
        if name == "gate" and self.words is not None:
            same = bool(np.array_equal(self.session.read(out)[0], self.words))
        ok = event.kind == "done" and wrong == 0 and same is not False
        return ok, wrong, relative, event.microseconds, same

    def timed(self, name, handler):
        """--calls commands of `handler`, each judged: ok, wrong, the median time."""
        calls = [self.call(name, handler) for _ in range(self.args.calls)]
        times = [call[3] for call in calls]
        hand.log(f"{name}: {times} us")
        record = {
            "ok": all(call[0] for call in calls),
            "wrong": sum(call[1] for call in calls),
            "max_relative": max(call[2] for call in calls),
            "microseconds": statistics.median(times),
            "all_microseconds": times,
        }
        if calls[0][4] is not None:  # every call's words against the hand program's
            record["hand_words_exact"] = all(call[4] for call in calls)
        return record

    def alternated(self, handlers):
        """gate then add, ALTERNATIONS times, each judged."""
        calls = [
            self.call(name, handlers[name])
            for _ in range(ALTERNATIONS)
            for name in ("gate", "add")
        ]
        return {
            "ok": all(call[0] for call in calls),
            "commands": len(calls),
            "wrong": sum(call[1] for call in calls),
        }


def tensors_of(session, args, cores):
    """a and b of the hand program's values (seed --seed), the two outputs, and the
    float64 reference of each handler."""
    pages = cores * args.tiles
    rng = np.random.default_rng(args.seed)
    a_bits, b_bits = hand.pages_of(rng, pages), hand.pages_of(rng, pages)
    a, b = (session.empty((pages, 1024), "bfloat16") for _ in range(2))
    for tensor, bits in ((a, a_bits), (b, b_bits)):
        session.write(tensor, bits.view(np.uint8).reshape(pages, -1))
    names = ("gate", "add", "hand")
    outs = {name: session.empty((pages, 1024), "bfloat16") for name in names}
    a_values, b_values = hand.floats(a_bits), hand.floats(b_bits)
    wanted = {"gate": hand.reference(a_values, b_values), "add": a_values + b_values}
    wanted["hand"] = wanted["gate"]
    return a, b, outs, wanted


def made(session, args, tensors):
    """The program (Mixed, or GateOnly and AddOnly merged) and its handlers."""
    a, b, outs, _ = tensors
    given = {"a": a, "b": b, "out": outs["gate"], "log": hand.log}
    if not args.merged:
        return Mixed(**given), {"gate": Mixed.gate, "add": Mixed.add}
    gate, add = GateOnly(**given), AddOnly(**given)
    merged = km.merge(gate, add, session=session, log=hand.log)
    return merged, {"gate": gate.gate, "add": add.add}


def hand_program(tensors, compute):
    """The gate's hand files as a program of their own (kernels/hand/): the class's
    pairs at its indices, its compute options (`compute`, the made class's)."""
    constants = km.abi.constants(HAND / "hand.h", known=km.abi.headers().constants)
    pages = constants["kPairAInPages"]
    buffers = [
        km.BufferPair(
            constants[f"kPair{name}MoverA"],
            constants[f"kPair{name}MoverB"],
            "bfloat16",
            hand.PAGE_BYTES,
            pages,
        )
        for name in ("AIn", "BIn", "Result")
    ]
    return km.Program(
        "gate_hand",
        kernels=HAND,  # movement.cpp for both movers, compute.cpp
        buffers=buffers,
        compile_args={
            "movement": [word for tensor in tensors for word in tensor.tensor_args()]
        },
        compute=dict(compute),
        handlers=HAND / "hand.h",
    )


def run_hand(session, args, cores, tensors, compute):
    """The hand program's GATE --calls times: its record and its output's words."""
    a, b, outs, _ = tensors
    program = hand_program((a, b, outs["hand"]), compute)
    run = session.load(program)
    try:
        record = Bench(session, run, tensors, args, cores).timed("hand", None)
        record["kernel_text"] = program.sizes.text
        record["compute"] = program.compute
    finally:
        run.stop()
    return record, session.read(outs["hand"])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pattern", default="AAAB", help="a pattern of A and B")
    parser.add_argument("--cores", type=int, default=12, help="0: the session's ranks")
    parser.add_argument("--tiles", type=int, default=64, help="tiles a core takes")
    parser.add_argument("--calls", type=int, default=5, help="timed commands a handler")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--merged", action="store_true", help="two classes, km.merge")
    parser.add_argument(
        "--add-alone", action="store_true", help="also add in a class of its own"
    )
    parser.add_argument(
        "--hand", action="store_true", help="first the gate's files as a hand program"
    )
    args = parser.parse_args()
    args.pattern = km.Pattern(args.pattern)
    block = Mixed.BLOCK.default
    if args.tiles % block or args.tiles < block:
        parser.error(f"--tiles is a multiple of {block}")
    report = {"form": "merged" if args.merged else "class"}
    with km.Session(log=hand.log) as session:
        cores = args.cores or session.ranks  # the grid less a Tensix dispatcher's
        report |= {"cores": cores, "tiles": args.tiles, "calls": args.calls}
        tensors = tensors_of(session, args, cores)
        program, handlers = made(session, args, tensors)
        words = None
        if args.hand:
            report["hand"], words = run_hand(
                session, args, cores, tensors, program.compute
            )
        run = session.load(program)
        try:
            bench = Bench(session, run, tensors, args, cores, words)
            report["kernel_text"] = program.sizes.text
            report["gate"] = bench.timed("gate", handlers["gate"])
            report["add"] = bench.timed("add", handlers["add"])
            report["alternate"] = bench.alternated(handlers)
        finally:
            run.stop()
        if args.add_alone:
            alone = AddOnly(a=tensors[0], b=tensors[1], out=tensors[2]["add"])
            run = session.load(alone)
            try:
                bench = Bench(session, run, tensors, args, cores)
                report["add_alone"] = bench.timed("add", AddOnly.add)
            finally:
                run.stop()
    report["ok"] = all(
        report[key]["ok"]
        for key in ("hand", "gate", "add", "alternate", "add_alone")
        if key in report
    )
    print(json.dumps(power.clocked(report), default=float), flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
