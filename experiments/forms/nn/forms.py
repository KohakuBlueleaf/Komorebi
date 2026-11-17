"""What every form of this folder shares: the log, tensors of tiles made from numpy
arrays and read back as arrays, the error of a result against its numpy reference, and
the JSON line a script ends with."""

import json
import sys
import time

import numpy as np

from komorebi import formats

STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def bf16(values):
    """`values` rounded to bfloat16, as float64: what the card holds of them."""
    bits = formats.bf16_bits(np.asarray(values, np.float32))
    return formats.bf16_values(bits).astype(np.float64)


def random(rng, shape, scale=1.0):
    """Normal values of `shape` times `scale`, rounded to bfloat16 (float64)."""
    return bf16(rng.standard_normal(shape) * scale)


def tiles(session, values, dtype="bfloat16"):
    """A tile-layout tensor holding the 2-D array `values` (written)."""
    return session.tensor(np.asarray(values, np.float32), dtype, "tile")


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


def timed(run, handler, group, args, timeout=30.0):
    """Submits one command and waits: its event's microseconds; raises unless done."""
    event = run.wait(run.submit(handler, group, args=args), timeout=timeout)
    if event.kind != "done":
        raise RuntimeError(f"{handler.name}: {event}")
    return round(event.microseconds, 1)


def finish(report, checks):
    """Prints the report with `ok` (every check's error under its bound) and exits."""
    report["ok"] = all(report[name][kind] <= bound for name, kind, bound in checks)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["ok"] else 1)
