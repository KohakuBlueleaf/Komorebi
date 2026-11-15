"""The gate of ../gate.py written as a class: out = silu(a) * b, the arithmetic a tile
function the tile compiler emits for compute, the streams and the handler compiled
into the program's files (build/gate-<hash>/ beside this file). See README.md.
Example: python gate_class.py --beside --cores 12
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import power

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import gate as hand  # noqa: E402  (the hand program, its values and its check)

SPANS_EMPTY, SPANS_FIRST = 16, 18  # a stamps page: the empty spans, then each call's
TOLERANCE = 1e-6  # the gate's largest relative error against float64 (fp32 values)


@km.tile_function(tolerance=TOLERANCE)
def silu_gate(a, b):
    """The gate of two tiles: what the tile compiler emits for compute."""
    return km.silu(a) * b


class Gate(km.Program):
    """gate: each core's `tiles` pages of a and b from rank x tiles, silu(a) * b into
    the same pages of out, block by block in the command's pattern."""

    BLOCK = km.constant(8)  # tiles a block, as the hand program's kBlock

    @km.handler
    def gate(
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
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(silu_gate(x, g))


class GateTimed(km.Program):
    """Gate's handler with stamps: the math CPU's cycles of each call of the tile
    function, to page rank of stamps."""

    BLOCK = km.constant(8)  # tiles a block, as Gate's

    @km.handler(stamps="stamps")
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        stamps: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=self.BLOCK, pattern=pattern
        )
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=self.BLOCK, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=self.BLOCK, pattern=pattern),
        ):
            results.push(silu_gate(x, g))


class GateBlocks(km.Program):
    """Gate's handler with its block a constant parameter: each block the make lists
    (handlers={"gate": [{"block": 8}, {"block": 4}]}) a handler of its own, the
    submit's value at its place picking it."""

    @km.handler
    def gate(
        self,
        a: km.Tensor,
        b: km.Tensor,
        out: km.Tensor,
        tiles: int,
        pattern: km.Pattern,
        block: km.constant[int] = 8,
    ):
        first = self.rank * tiles
        results = km.Stream(
            out.pages(first, tiles), write=True, block=block, pattern=pattern
        )
        for x, g in zip(
            km.Stream(a.pages(first, tiles), block=block, pattern=pattern),
            km.Stream(b.pages(first, tiles), block=block, pattern=pattern),
        ):
            results.push(silu_gate(x, g))


def values(rng, pages, dtype):
    """(the raw pages, their values in float64) of `pages` tiles: bfloat16 normal
    numbers as the hand program makes them, or float32 normal numbers times 8."""
    if dtype == "bfloat16":
        bits = hand.pages_of(rng, pages)
        return bits, hand.floats(bits)
    words = (rng.standard_normal((pages, 1024)) * 8).astype(np.float32)
    return words, words.astype(np.float64)


def compared(session, out, wanted, dtype):
    """The output against float64: the hand program's figures (largest error, largest
    relative error over values of 2^-4 or more, values outside its bound) and the
    largest relative error over every value whose reference is above 1e-30."""
    error = scale = relative = whole = 0.0
    wrong = 0
    for first, got in session.read_chunks(out):
        if dtype == "bfloat16":
            got = hand.floats(got.view(np.uint16).reshape(len(got), -1))
        else:
            got = got.view(np.float32).reshape(len(got), -1).astype(np.float64)
        want = wanted[first : first + len(got)]
        difference = np.abs(got - want)
        error = max(error, float(difference.max()))
        scale = max(scale, float(np.abs(want).max()))
        large = np.abs(want) >= hand.SMALL
        relative = max(relative, float((difference / np.abs(want))[large].max()))
        judged = np.abs(want) > 1e-30  # as experiments/language/tile_rules judges
        whole = max(whole, float((difference / np.abs(want))[judged].max()))
        wrong += int((difference > hand.BOUND * np.abs(want) + 2.0**-10).sum())
        hand.log(f"  output pages from {first} read back")
    return {
        "max_error": error,
        "scale": scale,
        "max_relative": relative,
        "max_relative_all": whole,
        "wrong": wrong,
    }


def spans(session, stamps, cores, tiles):
    """The math CPU's cycles a call from the stamps pages: each page's sequence and
    calls, the median empty span, the calls' median less it."""
    (raw,) = session.read(stamps)
    pages = raw.view(np.uint32).reshape(cores, -1).astype(np.int64)
    empty = float(np.median(pages[:, 2 : 2 + SPANS_EMPTY]))
    calls = pages[:, SPANS_FIRST : SPANS_FIRST + tiles]
    return {
        "calls": pages[:, 1].tolist() if cores <= 4 else int(pages[:, 1].sum()),
        "calls_ok": bool((pages[:, 1] == tiles).all()),
        "empty_span": empty,
        "median_call": float(np.median(calls)),
        "median_less_empty": float(np.median(calls)) - empty,
        "min_call": int(calls.min()),
        "max_call": int(calls.max()),
    }


def run_class(session, args, cores, dtype, timed):
    """The class made with tensors of `dtype` (`timed`: GateTimed, its stamps read),
    one command to `cores` cores, its output against float64."""
    pages = cores * args.tiles
    rng = np.random.default_rng(args.seed)
    a_raw, a_values = values(rng, pages, dtype)
    b_raw, b_values = values(rng, pages, dtype)
    a = session.empty((pages, 1024), dtype)
    b = session.empty((pages, 1024), dtype)
    out = session.empty((pages, 1024), dtype)
    stamps = session.empty((cores, 256), "uint32")
    for tensor, raw in ((a, a_raw), (b, b_raw)):
        session.write(tensor, raw.view(np.uint8).reshape(pages, -1))
    name = f"{'GateTimed' if timed else 'Gate'} {dtype}"
    hand.log(f"{name}: tensors written")
    made = GateTimed if timed else Gate
    program = made(
        a=a, b=b, out=out, stamps=stamps, log=hand.log, guard=not args.no_guard
    )
    run = session.load(program)
    try:
        words = (a, b, out, *([stamps] if timed else []), args.tiles, args.pattern)
        event = run.wait(run.submit(made.gate, (0, cores), args=words), timeout=30.0)
        hand.log(f"{name}: {event}")
        record = {"kernel_text": program.sizes.text, "event": event.kind}
        record["microseconds"] = event.microseconds
        record |= compared(session, out, hand.reference(a_values, b_values), dtype)
        if timed:
            record["spans"] = spans(session, stamps, cores, args.tiles)
    finally:
        run.stop()
        hand.log(f"{name}: run stopped")
    return record


def run_blocks(session, args, cores, blocks):
    """GateBlocks made with each of `blocks` listed (bfloat16), one load; a command
    a block, its output cleared before and judged after, as run_class's."""
    pages = cores * args.tiles
    rng = np.random.default_rng(args.seed)
    a_raw, a_values = values(rng, pages, "bfloat16")
    b_raw, b_values = values(rng, pages, "bfloat16")
    a, b, out = (session.empty((pages, 1024), "bfloat16") for _ in range(3))
    for tensor, raw in ((a, a_raw), (b, b_raw)):
        session.write(tensor, raw.view(np.uint8).reshape(pages, -1))
    listed = [{"block": block} for block in blocks]
    program = GateBlocks(a=a, b=b, out=out, log=hand.log, handlers={"gate": listed})
    run = session.load(program)
    records = {}
    try:
        for block in blocks:
            session.write(out, np.zeros((pages, 2048), np.uint8))
            words = (a, b, out, args.tiles, args.pattern, block)
            event = run.wait(run.submit(program.gate, (0, cores), args=words), 30.0)
            hand.log(f"GateBlocks block {block}: {event}")
            record = {"kernel_text": program.sizes.text, "event": event.kind}
            record["microseconds"] = event.microseconds
            record |= compared(
                session, out, hand.reference(a_values, b_values), "bfloat16"
            )
            records[f"block {block}"] = record
    finally:
        run.stop()
        hand.log("GateBlocks: run stopped")
    return records


def run_first_use(session, args, cores):
    """GateBlocks with block 8 listed and km.ANY (buffers sized for --reserve): 8, then
    each of --first-use (compiled or found built, added), --repeat commands each, each
    output judged; then each block listed in a load of its own (--listed-opted: also
    in one made with km.ANY), for the same commands' times."""
    pages = cores * args.tiles
    rng = np.random.default_rng(args.seed)
    a_raw, a_values = values(rng, pages, "bfloat16")
    b_raw, b_values = values(rng, pages, "bfloat16")
    a, b, out = (session.empty((pages, 1024), "bfloat16") for _ in range(3))
    for tensor, raw in ((a, a_raw), (b, b_raw)):
        session.write(tensor, raw.view(np.uint8).reshape(pages, -1))
    wanted = hand.reference(a_values, b_values)
    blocks = [Gate.BLOCK.default, *args.first_use]
    first_use = km.ANY(block=args.reserve) if args.reserve else km.ANY
    loads = [("first use", [{"block": blocks[0]}, first_use], blocks)]
    loads += [(f"listed {b}", [{"block": b}], [b]) for b in args.first_use]
    if args.listed_opted:  # the same, its kernels with the added handlers' path
        loads += [
            (f"listed {b}, km.ANY", [{"block": b}, km.ANY], [b]) for b in args.first_use
        ]
    records = {}
    for name, listed, submitted in loads:
        program = GateBlocks(a=a, b=b, out=out, log=hand.log, handlers={"gate": listed})
        run = session.load(program)
        try:
            for block in submitted:
                record = {"kernel_text": program.sizes.text, "runs": []}
                for _ in range(args.repeat):
                    session.write(out, np.zeros((pages, 2048), np.uint8))
                    words = (a, b, out, args.tiles, args.pattern, block)
                    started = time.perf_counter()
                    handle = run.submit(program.gate, (0, cores), args=words)
                    submitted_s = time.perf_counter() - started
                    event = run.wait(handle, 30.0)
                    judged = compared(session, out, wanted, "bfloat16")
                    record["runs"].append(
                        {
                            "event": event.kind,
                            "microseconds": event.microseconds,
                            "submit_s": submitted_s,
                            "wrong": judged["wrong"],
                            "max_relative": judged["max_relative"],
                        }
                    )
                record |= first_use_costs(program, run, block)
                record["event"] = "done"
                record["wrong"] = 0
                for one in record["runs"]:
                    record["wrong"] += one["wrong"]
                    if one["event"] != "done":
                        record["event"] = one["event"]
                hand.log(f"GateBlocks {name}, block {block}: {record}")
                records[f"{name}, block {block}"] = record
        finally:
            run.stop()
            hand.log(f"GateBlocks {name}: run stopped")
    return records


def first_use_costs(program, run, block):
    """What the first use of `block` took on the host: the class compile (or the look
    in the build folder), the parts' build (or the look in their cache), the parts'
    write and the entry commands; nothing for a listed block."""
    kept = getattr(program, "first_use", None)
    made = kept.made.get(("gate", (("block", block),))) if kept else None
    if made is None:
        return {}
    folder, variant, seconds, cached = made
    added = run.added[f"{kept.key}.{variant.name}"]
    return {
        "first_use": {
            "folder": folder.name,
            "compile_s": seconds,
            "compile_cached": cached,
            "number": added.number,
            "build_s": added.build_s,
            "build_cached": added.cached,
            "write_s": added.write_s,
            "entry_s": added.entry_s,
            "parts": added.parts,
        }
    }


def run_hand(session, args, cores):
    """../gate.py's program and command in this session, from the same seed."""
    pages = cores * args.tiles
    rng = np.random.default_rng(args.seed)
    a_bits, b_bits = hand.pages_of(rng, pages), hand.pages_of(rng, pages)
    a, b = session.tensor(a_bits, "uint16"), session.tensor(b_bits, "uint16")
    out = session.tensor(np.zeros_like(a_bits), "uint16")
    program = hand.program((a, b, out))
    run = session.load(program)
    try:
        words = [a.address, b.address, out.address, args.tiles, args.pattern.word]
        event = run.wait(run.submit("gate", (0, cores), words), timeout=30.0)
        hand.log(f"hand gate: {event}")
        record = {"kernel_text": program.sizes.text, "event": event.kind}
        record["microseconds"] = event.microseconds
        wanted = hand.reference(hand.floats(a_bits), hand.floats(b_bits))
        record |= compared(session, out, wanted, "bfloat16")
    finally:
        run.stop()
        hand.log("hand: run stopped")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--beside", action="store_true", help="the hand program first")
    parser.add_argument("--pattern", default="AAAB", help="a pattern of A and B")
    parser.add_argument("--cores", type=int, default=0, help="0: the whole grid")
    parser.add_argument("--tiles", type=int, default=64, help="tiles a core takes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-guard",
        action="store_true",
        help="the class made with guard=False (no give-up at a deadline, no inspect)",
    )
    parser.add_argument(
        "--formats",
        nargs="*",
        default=["bfloat16", "float32"],
        help="the class's tensors, each with Gate and GateTimed",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=[],
        help="GateBlocks with these blocks listed (bfloat16), a command each",
    )
    parser.add_argument(
        "--first-use",
        type=int,
        nargs="*",
        default=[],
        help="GateBlocks with block 8 listed and these blocks by first use",
    )
    parser.add_argument(
        "--reserve",
        type=int,
        default=0,
        help="first use: the load's buffers also sized for this block (km.ANY(block=))",
    )
    parser.add_argument("--repeat", type=int, default=5, help="first use: commands")
    parser.add_argument(
        "--listed-opted",
        action="store_true",
        help="first use: each listed block also in a load made with km.ANY",
    )
    args = parser.parse_args()
    args.pattern = km.Pattern(args.pattern)
    block = Gate.BLOCK.default
    if args.tiles % block or not block <= args.tiles <= 256 - SPANS_FIRST:
        parser.error(f"--tiles is a multiple of {block} up to {256 - SPANS_FIRST}")
    if any(block < 1 or args.tiles % block for block in args.blocks + args.first_use):
        parser.error("each of --blocks and --first-use divides --tiles")
    report = {"runs": {}}
    with km.Session(log=hand.log) as session:
        cores = args.cores or session.grid[0] * session.grid[1]
        hand.log(f"device open; {cores} cores, {args.tiles} tiles each")
        if args.beside:
            report["runs"]["hand"] = run_hand(session, args, cores)
        for dtype in args.formats:
            for timed in (False, True):
                record = run_class(session, args, cores, dtype, timed)
                report["runs"][f"{'timed ' if timed else ''}class {dtype}"] = record
        if args.blocks:
            for name, record in run_blocks(session, args, cores, args.blocks).items():
                report["runs"][f"class bfloat16 {name}"] = record
        if args.first_use:
            for name, record in run_first_use(session, args, cores).items():
                report["runs"][f"class bfloat16 {name}"] = record
    ok = True
    for name, record in report["runs"].items():
        ok &= record["event"] == "done" and record["wrong"] == 0
        if name.endswith("float32"):
            ok &= record["max_relative_all"] <= TOLERANCE
        if "spans" in record:
            ok &= record["spans"]["calls_ok"]
    report["ok"] = bool(ok)
    print(json.dumps(power.clocked(report), default=float), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
