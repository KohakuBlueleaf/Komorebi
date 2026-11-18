"""What every form of this folder shares: the log, tensors of tiles made from numpy
arrays and read back as arrays, the error of a result against its numpy reference, and
the JSON line a script ends with."""

import json
import sys
import time

import numpy as np

import komorebi as km
from komorebi import formats

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def card_options(parser):
    parser.add_argument("--cards", default="0", help="0, or 0,1: two cards as islands")
    parser.add_argument(
        "--place",
        choices=("edge", "inside", "own"),
        default="edge",
        help="two cards: groups across the edge with tensors where the session puts "
        "them (edge), groups and tensors on island 0 (inside), groups across the "
        "edge with each tensor split over its group (own)",
    )


def open_session(args):
    """One card's session, or with --cards 0,1 the two cards as islands."""
    cards = [int(card) for card in args.cards.split(",")]
    return km.Session(cards=cards, log=log) if len(cards) > 1 else km.Session(log=log)


def group(session, args, count):
    """(first, count): from rank 0 on one card; on two, across the islands' edge,
    half the group on each island."""
    if "," not in args.cards or args.place == "inside":
        return (0, count)
    return (session.graph.first(1) - count // 2, count)


def placed(session, args, count):
    """session.tensor's hints for a tensor a group of `count` ranks reads: its pages
    split over that group (--place inside or own), else none."""
    if "," not in args.cards or args.place == "edge":
        return {}
    return {"split": 0, "group": group(session, args, count)}


def copied(args):
    """The hint for a small tensor every rank reads: a copy on every island."""
    if "," not in args.cards or args.place == "edge":
        return {}
    return {"copy": True}


def bf16(values):
    """`values` rounded to bfloat16, as float64: what the card holds of them."""
    bits = formats.bf16_bits(np.asarray(values, np.float32))
    return formats.bf16_values(bits).astype(np.float64)


def random(rng, shape, scale=1.0):
    """Normal values of `shape` times `scale`, rounded to bfloat16 (float64)."""
    return bf16(rng.standard_normal(shape) * scale)


def tiles(session, values, dtype="bfloat16", **hints):
    """A tile-layout tensor holding the 2-D array `values` (written); `hints`
    place it on islands (split=, group=, copy=)."""
    return session.tensor(np.asarray(values, np.float32), dtype, "tile", **hints)


def empty(session, rows, columns, dtype="bfloat16"):
    return session.empty((rows, columns), dtype, "tile")


def read(session, tensor):
    """A tile-layout tensor's values as a float64 [rows, columns] array."""
    (raw,) = session.read(tensor)
    rows, columns = tensor.shape[-2:]
    return formats.from_bytes(raw, (rows, columns), tensor.dtype, "tile").astype(
        np.float64
    )


def words(session, tensor):
    """A tensor's raw pages as uint32 words, [pages, words a page]."""
    (raw,) = session.read(tensor)
    return np.ascontiguousarray(raw).view(np.uint32)


def error(got, want):
    """rms and largest error of `got` against `want`, each over the rms of `want`."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = float(np.sqrt(np.mean(want * want))) or 1.0
    diff = got - want
    return {
        "rms": float(np.sqrt(np.mean(diff * diff))) / scale,
        "max": float(np.max(np.abs(diff))) / scale,
    }


def timed(run, handler, group, args, timeout=30.0, across=None):
    """Submits one command and waits: its event's microseconds; raises unless done.
    A list `across` gets what crossed the links meanwhile (an island run's)."""
    before = run.crossings() if across is not None else None
    event = run.wait(run.submit(handler, group, args=args), timeout=timeout)
    if event.kind != "done":
        raise RuntimeError(f"{handler.name}: {event}")
    if across is not None:
        total = (run.crossings() - before).total()
        kept = {k: total[k] for k in ("lines", "asks", "pages", "puts", "bytes")}
        across.append({"command": handler.name, **kept})
    return round(event.microseconds, 1)


def across_list(args):
    """A list for timed's `across` on two cards, else None."""
    return [] if "," in args.cards else None


def finish(report, checks):
    """Prints the report with `ok` (every check's error under its bound) and exits."""
    report["ok"] = all(report[name][kind] <= bound for name, kind, bound in checks)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)
