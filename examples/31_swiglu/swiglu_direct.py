"""The SwiGLU chain with the hidden sent direct (km.direct) into a landing area or into
down's stream halves (in_place), for the class as printed and a reader-heavy shape:
each form's y word for word, its time and its stamps' terms by record, and the hidden's
host read refused after a direct chain. See README.md."""

import argparse
import json
import sys
import time

import numpy as np
from swiglu_class import SwiGLU, errors, gate, inputs, tensors_of, wanted

import komorebi as km
from komorebi import power
from komorebi.dsl import chain_stamps

STARTED = time.monotonic()
NAMES = ("x", "w1", "w3", "w2", "hidden", "y")


class DirectSwiGLU(SwiGLU):
    """SwiGLU whose down may take the hidden direct, and the chain that sends it."""

    down = km.handler(direct=("hidden",))(SwiGLU.down.function)

    @km.chain
    def forward_direct(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(km.direct(hidden), w2, y, cores=(0, 4))  # up's blocks into L1


class InPlaceSwiGLU(DirectSwiGLU):
    """... the hidden landing in down's stream halves (no copy)."""

    down = km.handler(direct=("hidden",), in_place=True)(SwiGLU.down.function)


class WideSwiGLU(km.Program):
    """The reader-heavy shape: up's 8 columns a core in 2 pieces of 4 (each written
    when its K loop ends), down on 32 cores of one column (the hidden half of each
    reader's reads)."""

    D = km.constant(32)
    H = km.constant(96)
    WIDTH = km.constant(8)  # up's columns a core
    PIECE = km.constant(4)  # ... written at once
    CHUNK = km.constant(2)  # K tiles a block
    W_PIECE = CHUNK * PIECE

    h0: km.L1[km.fp32, 1, PIECE, 32, 32]
    g0: km.L1[km.fp32, 1, PIECE, 32, 32]
    h1: km.L1[km.fp32, 1, PIECE, 32, 32]
    g1: km.L1[km.fp32, 1, PIECE, 32, 32]
    out: km.L1[km.fp32, 1, 1, 32, 32]

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def up(self, x: km.Tensor, w1: km.Tensor, w3: km.Tensor, hidden: km.Tensor):
        first = self.rank * self.WIDTH
        for xb, w1b, w3b in zip(
            km.Stream(x.tiles[0:1, 0 : self.D], block=self.CHUNK),
            km.Stream(
                w1.tiles[0 : self.D, first : first + self.PIECE], block=self.W_PIECE
            ),
            km.Stream(
                w3.tiles[0 : self.D, first : first + self.PIECE], block=self.W_PIECE
            ),
        ):
            self.h0 += self.project(xb, w1b)
            self.g0 += self.project(xb, w3b)
        hidden.tiles[0:1, first : first + self.PIECE].write(gate(self.h0, self.g0))
        second = first + self.PIECE
        for xb, w1b, w3b in zip(
            km.Stream(x.tiles[0:1, 0 : self.D], block=self.CHUNK),
            km.Stream(
                w1.tiles[0 : self.D, second : second + self.PIECE], block=self.W_PIECE
            ),
            km.Stream(
                w3.tiles[0 : self.D, second : second + self.PIECE], block=self.W_PIECE
            ),
        ):
            self.h1 += self.project(xb, w1b)
            self.g1 += self.project(xb, w3b)
        hidden.tiles[0:1, second : second + self.PIECE].write(gate(self.h1, self.g1))

    @km.handler
    def down(self, hidden: km.Tensor, w2: km.Tensor, y: km.Tensor):
        first = self.rank  # this core's column of y
        for hb, wb in zip(
            km.Stream(hidden.tiles[0:1, 0 : self.H], block=self.CHUNK),
            km.Stream(w2.tiles[0 : self.H, first : first + 1], block=self.CHUNK),
        ):
            self.out += self.project(hb, wb)
        y.tiles[0:1, first : first + 1].write(self.out)

    @km.chain
    def forward(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(hidden, w2, y, cores=(0, 32))


class WideDirect(WideSwiGLU):
    down = km.handler(direct=("hidden",))(WideSwiGLU.down.function)

    @km.chain
    def forward_direct(self, x, w1, w3, w2, hidden, y):
        self.up(x, w1, w3, hidden, cores=(0, 12))
        self.down(km.direct(hidden), w2, y, cores=(0, 32))


class WideInPlace(WideDirect):
    down = km.handler(direct=("hidden",), in_place=True)(WideSwiGLU.down.function)


# Each shape: its plain class, its direct one (a copy), its in-place one; its cores.
LINES = {
    "example": (SwiGLU, DirectSwiGLU, InPlaceSwiGLU, 12),
    "wide": (WideSwiGLU, WideDirect, WideInPlace, 32),
}


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=50, help="timed chains a form")
    parser.add_argument("--stamped", type=int, default=20, help="stamped chains")
    parser.add_argument(
        "--deadline", type=float, default=0.5, help="a command's deadline, s"
    )
    parser.add_argument("--lines", default="example,wide", help="the shapes run")
    parser.add_argument(
        "--kernel-bytes",
        type=int,
        default=None,
        help="the session's kernel buffer, the largest form's (a card is opened "
        "again for larger kernels only before its first run)",
    )
    return parser.parse_args()


def chain_of(run, program, t, form):
    method = program.forward if form == "dram" else program.forward_direct
    return run.chain(method(*(t[n] for n in NAMES)))


def once(session, program, t, form, args):
    """A load, the chain `form` once, the stop: (y, the run's error, the hidden's host
    read refused or None, the chain's report)."""
    t["hidden"].write(np.zeros((32, 96 * 32), np.float32))
    t["y"].write(np.zeros((32, 32 * 32), np.float32))
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    chain = chain_of(run, program, t, form)
    try:
        event = run.wait(run.submit(chain))
    finally:
        run.stop()
    if event.kind != "done":
        raise RuntimeError(f"{type(program).__name__} {form} chain: {event}")
    try:
        t["hidden"].read()
        refused = None
    except RuntimeError as refusal:
        refused = str(refusal)
    return t["y"].read(), run.result()["error_name"], refused, chain.report


def timed(session, program, t, forms, args):
    """A load, the chains `forms` in turn `runs` times each: their card us, median;
    the run's error."""
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    chains = {form: chain_of(run, program, t, form) for form in forms}
    times = {form: [] for form in chains}
    try:
        for _ in range(1 + args.runs):  # the first of each not timed
            for form, chain in chains.items():
                event = run.wait(run.submit(chain))
                if event.kind != "done":
                    raise RuntimeError(f"{form} chain: {event}")
                times[form].append(event.microseconds)
    finally:
        run.stop()
    error = run.result()["error_name"]
    return {f: round(float(np.median(v[1:])), 2) for f, v in times.items()}, error


def stamped(session, cls, t, cores, args):
    """`cls` made with the chain's stamps, a load: each chain once (its stamps read and
    dropped), then `stamped` times and read: each record's terms (us) by form."""
    program = cls(log=log, session=session, chain_stamps=True, **t)
    run = session.load(program, wait=args.deadline, deadline=args.deadline)
    rows_tensor = chain_stamps.tensor_for(session, cores)
    mhz = power.aiclk_mhz()
    out = {}
    try:
        for form in ("dram", "direct"):
            chain = chain_of(run, program, t, form)
            for count in (1, args.stamped):
                for _ in range(count):
                    event = run.wait(run.submit(chain))
                    if event.kind != "done":
                        raise RuntimeError(f"stamped {form} chain: {event}")
                rows = chain_stamps.read(session, run, program, rows_tensor, (0, cores))
            out[form] = chain_stamps.terms(rows, mhz, 2)
    finally:
        run.stop()
    out["kernel"] = run.result()["error_name"]
    return out


def line_of(session, data, t, name, args):
    """One shape: the DRAM chain, the direct chain and the in-place chain each once
    (y word for word), then timed, then stamped."""
    plain, copy, placed, cores = LINES[name]
    result = {"kernel": []}
    ys, refused, reports = {}, {}, {}
    for cls, forms in ((copy, ("dram", "direct")), (placed, ("dram", "direct"))):
        program = cls(log=log, session=session, **t)
        for form in forms:
            key = form if form == "dram" else f"{form}_{cls.__name__}"
            y, error, refusal, report = once(session, program, t, form, args)
            if form == "dram" and "dram" in ys:
                key = f"dram_{cls.__name__}"
            ys[key], refused[key] = y, refusal
            reports[key] = {str(k): len(v) for k, v in report.items()}
            result["kernel"].append(error)
            if form == "dram" and "hidden_dram" not in result:
                result["hidden_dram"] = t["hidden"].read()
    want_y = wanted(data, result.pop("hidden_dram"))[2]
    result["y_words_differ"] = {
        key: int(np.sum(y != ys["dram"])) for key, y in ys.items()
    }
    result["y_err"] = {k: errors(y.astype(np.float64), want_y) for k, y in ys.items()}
    result["hidden_read_refused"] = {k: v is not None for k, v in refused.items()}
    result["roots"] = reports
    right = all(e == "none" for e in result["kernel"]) and not any(
        result["y_words_differ"].values()
    )
    if not right:
        return result
    us = {}
    times, error = timed(
        session, plain(log=log, session=session, **t), t, ["dram"], args
    )
    us[f"dram_{plain.__name__}"] = times["dram"]
    result["kernel"].append(error)
    for cls in (copy, placed):
        program = cls(log=log, session=session, **t)
        times, error = timed(session, program, t, ["dram", "direct"], args)
        us[f"dram_{cls.__name__}"] = times["dram"]
        us[f"direct_{cls.__name__}"] = times["direct"]
        result["kernel"].append(error)
    result["card_us"] = us
    result["terms"] = {}
    for cls in (copy, placed):
        terms = stamped(session, cls, t, cores, args)
        result["kernel"].append(terms.pop("kernel"))
        result["terms"][cls.__name__] = terms
    return result


def main():
    args = parse()
    data = inputs()
    result = {"shape": [32, 1024, 3072]}
    with km.Session(log=log, kernel_bytes=args.kernel_bytes) as session:
        t = tensors_of(session, data, empty=False)
        for name in args.lines.split(","):
            result[name] = line_of(session, data, t, name, args)
        power.clocked(result)  # in the session: the clock the times were taken at
    result["ok"] = all(
        "card_us" in result[name] and all(e == "none" for e in result[name]["kernel"])
        for name in args.lines.split(",")
    )
    result["wall_s"] = round(time.monotonic() - STARTED, 1)
    print(json.dumps(power.clocked(result)), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
