"""One apply y = M x of a set's hierarchical transport matrix on the card, as one
command of the Transport class on every core: checked against the host's emulation and
the set's y_q8.npy and y_float.npy, timed over --calls calls, and run --again times
more with y fed back as x. Prints one JSON line. Usage in README.md."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import formats  # noqa: E402
from komorebi.bench import power  # noqa: E402
from projects.transport import plan as pl  # noqa: E402
from projects.transport.transport import Transport  # noqa: E402

LIT = 1e-4  # an element's luminance counted in the errors (the host step's)
LUMA = np.array([0.2126, 0.7152, 0.0722])
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def errors(got, want, lit):
    """Per-element relative error of the luminance over the lit elements: median,
    99th percentile, most; and the ratio of the luminance sums."""
    g, w = got @ LUMA, want @ LUMA
    rel = np.abs(g[lit] - w[lit]) / w[lit]
    return {
        "median": float(np.median(rel)),
        "p99": float(np.percentile(rel, 99)),
        "most": float(rel.max()),
        "sum_ratio": float(g.sum() / w.sum()),
    }


def float_apply(s, x):
    """The set's float32 apply (its README's loop): y [N, 3]."""
    numbers = s.weights()
    y = np.zeros_like(x)
    for q in range(s.blocks):
        r0, r1 = s.row0[q] * pl.TILE, s.row1[q] * pl.TILE
        c0, c1 = s.col0[q] * pl.TILE, s.col1[q] * pl.TILE
        m, n, xr, a = r1 - r0, c1 - c0, x[c0:c1].T, int(s.offset_a[q])
        kp = int(s.rank[q]) * pl.TILE
        if kp == 0:
            y[r0:r1] += (xr @ numbers[a : a + n * m].reshape(n, m)).T
        else:
            b = int(s.offset_b[q])
            v = numbers[a : a + n * kp].reshape(n, kp)
            y[r0:r1] += ((xr @ v) @ numbers[b : b + kp * m].reshape(kp, m)).T
    return y


def y_of(session, y, tiles):
    """y's pages read while the run is loaded, as [N, 3] (rows 0 to 2 of its tiles)."""
    (raw,) = session.read(y)
    values = np.ascontiguousarray(raw).view(np.float32).reshape(tiles, pl.TILE**2)
    return formats.untilize(values, pl.TILE, pl.TILE * tiles)[:3].T.copy()


def x_pages(x):
    """x [N, 3] as the bfloat16 tile tensor's pages [tiles, 2048] uint8."""
    raw, _ = formats.to_bytes(pl.x_tiles(x), "bfloat16", "tile")
    return raw.reshape(-1, pl.X_TILE_BYTES)


def fit(columns, span):
    """Least squares span ~ columns @ c: the coefficients and the RMS residual."""
    a = np.stack(columns, axis=1).astype(np.float64)
    c, *_ = np.linalg.lstsq(a, span, rcond=None)
    return [float(v) for v in c], float(np.sqrt(np.mean((a @ c - span) ** 2)))


def per_core(t, clock, command_us):
    """Each core's unpack stamps of the last command (wall-clock cycles: span, busy in
    the product steps, start, end), its entries and reads; the slowest core; fits of
    the span to the weight tiles and the entries."""
    span, busy = clock[:, 0].astype(np.float64), clock[:, 1].astype(np.float64)
    start = clock[:, 2] - clock[:, 2].min()
    entries = np.array([len(r) for r in t.entries], np.float64)
    tiles = t.tiles_read.astype(np.float64)
    xs = t.x_read.astype(np.float64)
    slow = int(np.argmax(span))
    ones = np.ones_like(span)
    two, rms2 = fit([tiles, entries], span)
    three, rms3 = fit([tiles, entries, ones], span)
    four, rms4 = fit([tiles, xs, entries, ones], span)
    return {
        "span_cycles_most": float(span.max()),
        "command_us": command_us,
        "slowest": {
            "core": slow,
            "span": float(span[slow]),
            "busy": float(busy[slow]),
            "waits": float(span[slow] - busy[slow]),
            "start_after_first": int(start[slow]),
            "pack_span": int(clock[slow, 4]),
            "pack_busy": int(clock[slow, 5]),
            "entries": int(entries[slow]),
            "tiles": int(tiles[slow]),
            "x_tiles": int(xs[slow]),
            "weight_bytes": int(tiles[slow]) * pl.WEIGHT_TILE_BYTES,
            "x_bytes": int(xs[slow]) * pl.X_TILE_BYTES,
        },
        "start_spread": int(start.max()),
        "busy_mean_over_span_mean": float(busy.mean() / span.mean()),
        "fit_tiles_entries": {"a": two[0], "b": two[1], "rms": rms2},
        "fit_tiles_entries_1": {
            "a": three[0], "b": three[1], "c": three[2], "rms": rms3,
        },  # fmt: skip
        "fit_tiles_x_entries_1": {
            "a": four[0], "ax": four[1], "b": four[2], "c": four[3], "rms": rms4,
        },  # fmt: skip
        "rows": [
            [int(v) for v in row]
            for row in zip(span, busy, start, entries, tiles, xs, clock[:, 4])
        ],
    }


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="the set's files (blocks.npz, weights.f32, ...)")
    parser.add_argument("--cache", default=None, help="a folder for w's bytes")
    parser.add_argument("--cores", type=int, default=120)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--again", type=int, default=2, help="applies of y as x")
    parser.add_argument("--block", type=int, default=8, help="weight tiles a block")
    parser.add_argument(
        "--reverse", action="store_true", help="ranks take the ranges reversed"
    )
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main():
    args = parse()
    s = pl.load(args.folder)
    first = pl.deal(s, args.cores)
    _, pages = pl.weight_pages(s)
    t = pl.tables(s, first, pages - 1)
    if args.reverse:  # rank c takes the table of core cores - 1 - c (a diagnostic)
        t.words = t.words.reshape(args.cores, t.pages, -1)[::-1].reshape(t.words.shape)
        t.entries, t.tiles_read, t.x_read = (
            t.entries[::-1],
            t.tiles_read[::-1],
            t.x_read[::-1],
        )
    report = {"set": Path(args.folder).name, "sizes": pl.sizes(s, t)}
    log(f"sizes {json.dumps(report['sizes'])}")
    cache = Path(args.cache) / f"{Path(args.folder).name}_w.npy" if args.cache else None
    raw = pl.encode_weights(s, cache)
    log(f"w: {len(raw)} pages encoded")
    x = s.vector("x.npy")
    lit = s.vector("y_exact.npy") @ LUMA > LIT
    lit[np.load(Path(args.folder) / "perm.npy") < 0] = False
    emulated = pl.emulate(s, pl.decode_weights(raw), x)
    log("emulated")
    index = power.visible_card()
    with km.Session(log=log) as session:
        tx = session.tensor(pl.x_tiles(x), "bfloat16", "tile")
        tw = session.empty((pages, pl.TILE, pl.TILE), "bfp8_b", "tile")
        tw.transposed_tiles = True
        session.write(tw, np.asarray(raw))
        ty = session.empty((pl.TILE, pl.TILE * s.tiles), "float32", "tile")
        tt = session.tensor(t.words, "uint32")
        tp = session.empty((args.cores, 32), "uint32")
        sizes = report["sizes"]
        program = Transport(
            x=tx, w=tw, y=ty, tables=tt, profile=tp, session=session, log=log,
            OWN=sizes["own_most"], KMAX=sizes["k_most"], RANK=sizes["rank_most"],
            BLOCK=args.block,
        )  # fmt: skip
        run = session.load(program)
        group, words = (0, args.cores), (tx, tw, ty, tt, tp, t.pages)
        event = run.wait(run.submit(Transport.apply, group, args=words), timeout=60.0)
        if event.kind != "done":
            raise km.KomorebiError(f"apply: {event}")
        got = y_of(session, ty, s.tiles)
        log(f"first apply {event.microseconds:.1f} us")
        card_us, wall_us = [], []
        before = power.aiclk_mhz(index)
        for _ in range(args.calls):
            started = time.perf_counter()
            event = run.wait(run.submit(Transport.apply, group, args=words))
            wall_us.append((time.perf_counter() - started) * 1e6)
            if event.kind != "done":
                raise km.KomorebiError(f"apply: {event}")
            card_us.append(event.microseconds)
        report["time"] = {
            "calls": args.calls,
            "card_us_median": float(np.median(card_us)),
            "card_us_least": float(np.min(card_us)),
            "wall_us_median": float(np.median(wall_us)),
            "wall_us_least": float(np.min(wall_us)),
            "mhz": [before, power.aiclk_mhz(index)],
        }
        log(f"time {json.dumps(report['time'])}")
        again, previous = [], got
        for _ in range(args.again):
            session.write(tx, x_pages(previous))
            event = run.wait(run.submit(Transport.apply, group, args=words))
            if event.kind != "done":
                raise km.KomorebiError(f"apply again: {event}")
            now = y_of(session, ty, s.tiles)
            again.append((previous, now))
            previous = now
        (clock,) = session.read(tp)
        clock = np.ascontiguousarray(clock).view(np.uint32).astype(np.int64)
        run.stop()
        report["kernel"] = run.result()["error_name"]
    report["cores"] = per_core(t, clock, report["time"]["card_us_median"])
    report["check"] = {
        "emulated": errors(got, emulated, lit),
        "y_q8": errors(got, s.vector("y_q8.npy"), lit),
        "y_float": errors(got, s.vector("y_float.npy"), lit),
        "emulated_vs_y_q8": errors(emulated, s.vector("y_q8.npy"), lit),
        "emulated_vs_y_float": errors(emulated, s.vector("y_float.npy"), lit),
        "lit": int(lit.sum()),
    }
    w = pl.decode_weights(raw)
    report["again"] = []
    for k, (given, now) in enumerate(again):
        lit_k = float_apply(s, given) @ LUMA > LIT
        report["again"].append(
            {
                "emulated": errors(now, pl.emulate(s, w, given), lit_k),
                "float": errors(now, float_apply(s, given), lit_k),
                "lit": int(lit_k.sum()),
                "apply": k + 2,
            }
        )
    report["ok"] = report["kernel"] == "none"
    print(json.dumps(report, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
