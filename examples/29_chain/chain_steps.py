"""Four calls of a small program, each reading what the one before wrote, run as a
command a step and as one chain: every output exact against numpy, both ways timed,
an earlier input proven and one refused. One JSON line. See README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import chain as kc
from komorebi.bench import power

KERNELS = Path(__file__).resolve().parent / "kernels"
C = km.abi.constants(KERNELS / "steps.h", known=km.abi.headers().constants)
WORDS = C["kTileWords"]
CHAIN_PAGE = 2816  # the chain's words on each core (km_chain.hpp's kChainAreaBytes)
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cores", type=int, default=8, help="cores 0 .. N - 1")
    parser.add_argument("--width", type=int, default=2, help="tiles a core")
    parser.add_argument("--runs", type=int, default=200, help="timed runs each way")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--form",
        choices=["kernel", "runtime", "merged"],
        default="kernel",
        help="the program's form: its own kernels, or parts the runtime's kernels call "
        "(merged: SUM one program, SCALE and ADD another, merged into one)",
    )
    return parser.parse_args()


def program(tiles, form="kernel"):
    page = C["kTileBytes"]
    return km.Program(
        "chain_steps",
        kernels=KERNELS if form == "kernel" else KERNELS / "runtime",
        buffers=[
            km.Buffer(C["kCbIn"], "uint32", page, 1),
            km.Buffer(C["kCbOther"], "uint32", page, 1),
            km.Buffer(C["kCbArrive"], "uint32", page, C["kArrivePages"]),
            km.Buffer(C["kCbOut"], "uint32", page, C["kOutPages"]),
            km.Buffer(kc.words()["kChainBuffer"], "uint32", CHAIN_PAGE, 1),
        ],
        compile_args={"reader": tiles.tensor_args(), "writer": tiles.tensor_args()},
        handlers=KERNELS / "steps.h",
    )


def merged(tiles):
    """The runtime's form as two programs merged into one (km.merge): `sums` holds SUM
    (its 3 kept), `maps` SCALE and ADD (their 1 and 2 become 4 and 5); a chain's
    records name handlers of both, each taken to its source's own number."""
    sums, maps = program(tiles, "runtime"), program(tiles, "runtime")
    sums.name, sums.handlers = "sums", {"sum": C["kHandlerSum"]}
    maps.name = "maps"
    maps.handlers = {"scale": C["kHandlerScale"], "add": C["kHandlerAdd"]}
    return km.merge(sums, maps, log=log)


def calls(t, cores, width, add_read=None, first=0):
    """SCALE x by the per-submit word into y, SCALE y by 3 into z, ADD y to z into s
    (y an earlier input), SUM s's tiles to a root into t: core g's tiles g x width on.
    `add_read`: ADD's range of y (default its own tiles); `first`: the first core."""
    own = kc.Range(0, width, width)  # group g: tiles g x width .. + width
    tiles = cores * width
    group = (first, cores)
    add = kc.Read("y", ranges=(add_read or own,))
    return [
        kc.Call(
            "scale",
            (0, t["x"], t["y"], 0, 0, 0, width),
            group,
            1,
            width,
            "y",
            word=("factor", 5),
            name="scale x",
        ),
        kc.Call(
            "scale",
            (0, t["y"], t["z"], 0, 0, 3, width),
            group,
            1,
            width,
            "z",
            "y",
            ranges=(own,),
            name="scale y",
        ),
        kc.Call(
            "add",
            (0, t["z"], t["s"], t["y"], 0, 0, width),
            group,
            1,
            width,
            "s",
            "z",
            ranges=(own,),
            reads=(add,),
            name="add",
        ),
        kc.Call(
            "sum",
            (0, t["s"], t["t"], tiles, 0, 0, 1),
            group,
            cores,
            1,
            "t",
            "s",
            k=(0, tiles),
            sends=True,
            name="sum",
        ),
    ]


def by_steps(run, steps, factor):
    """A command a step, each waited for: the factor ORed into the first one's a5. The
    dispatcher's submit-to-done microseconds, summed."""
    total = 0.0
    for index, call in enumerate(steps):
        args = list(call.args)
        args[5] |= factor if index == 0 else 0
        handle = run.submit(call.handler, call.group, args, exclusive=True)
        total += run.wait(handle).microseconds
    return total


def by_chain(run, chain, factor):
    """The chain as one command, waited for: its submit-to-done microseconds."""
    return run.wait(run.submit(chain, factor=factor)).microseconds


def expected(x, factor):
    y = x * factor
    z = y * 3
    s = z + y
    return {"y": y, "z": z, "s": s, "t": s.sum(axis=0, dtype=np.int32)[None]}


def outputs(session, tensors):
    got = session.read(*(tensors[name] for name in ("y", "z", "s", "t")))
    return {
        name: raw.view(np.int32).reshape(-1, WORDS)
        for name, raw in zip(("y", "z", "s", "t"), got, strict=True)
    }


def bad_words(got, want):
    return {name: int(np.sum(got[name] != want[name])) for name in want}


def main():
    args = parse()
    generator = np.random.default_rng(args.seed)
    tiles = args.cores * args.width
    x = generator.integers(-50, 51, (tiles, WORDS)).astype(np.int32)
    result = {"cores": args.cores, "width": args.width, "tiles": tiles}
    result["form"] = args.form
    with km.Session(log=log) as session:
        tensors = {"x": session.tensor(x, "int32")}
        for name, rows in (("y", tiles), ("z", tiles), ("s", tiles), ("t", 1)):
            tensors[name] = session.tensor(np.zeros((rows, WORDS), np.int32), "int32")
        t = {name: tensor.address for name, tensor in tensors.items()}
        steps = calls(t, args.cores, args.width)
        try:
            shifted = kc.Range(args.width, args.width, args.width)  # the next core's
            kc.build(calls(t, args.cores, args.width, shifted))
            result["refused"] = None
        except kc.ChainRefused as error:
            result["refused"] = str(error)
        loaded = (
            merged(tensors["x"])
            if args.form == "merged"
            else program(tensors["x"], args.form)
        )
        run = session.load(loaded, wait=0.5, deadline=0.5)
        chain = run.chain(steps, C["kChainMode"])
        log(f"loaded; the chain's records written ({len(steps)})")
        checks = {}
        for way, factor in (("steps", 2), ("chain", 5), ("steps", 7), ("chain", 3)):
            if way == "steps":
                by_steps(run, steps, factor)
            else:
                by_chain(run, chain, factor)
            checks[f"{way} x{factor}"] = bad_words(
                outputs(session, tensors), expected(x, factor)
            )
        timed, card = {}, {}
        for way in ("steps", "chain"):
            start = time.perf_counter()
            times = [
                by_steps(run, steps, 1) if way == "steps" else by_chain(run, chain, 1)
                for _ in range(args.runs)
            ]
            timed[way] = round((time.perf_counter() - start) * 1e6 / args.runs, 2)
            card[way] = round(float(np.median(times)), 2)
        checks["after timing"] = bad_words(outputs(session, tensors), expected(x, 1))
        run.stop()
        chain.free()
        result["kernel"] = run.result()["error_name"]
    result["checks"] = checks
    result["microseconds_a_run"] = timed  # the host's, submit to the last wait
    result["card_microseconds_a_run"] = card  # the dispatcher's, median
    result["earlier_proven"] = True  # run.chain built the chain: ADD's y read proven
    result["ok"] = bool(
        result["kernel"] == "none"
        and all(not any(bad.values()) for bad in checks.values())
        and result["refused"] is not None
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
