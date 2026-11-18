"""One attention shape on the card, checked against float64 on sampled query tile rows
(their output tiles are read back, not the whole tensor). Prints one JSON line; exits 1
when the check fails. Usage in README.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi.bench import power  # noqa: E402
from projects.attention import card  # noqa: E402
from projects.attention import plan as pl  # noqa: E402
from projects.attention.reference import sdpa  # noqa: E402

# Bound of the check on standard normal inputs (README.md, "The check's bound"); the
# approximate exponential's relative RMS bound.
RMS_BOUND, MAX_BOUND = 1.5e-2, 6e-2
APPROX_RMS_BOUND = 3e-2
SAMPLED = 16  # query tile rows checked
STARTED = time.monotonic()


def log(message):
    """A progress line with seconds since start, flushed."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shape", choices=pl.SHAPES, default="t1")
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument(
        "--broadcast", type=int, default=None, help="broadcast length (default: plan)"
    )
    parser.add_argument(
        "--chunks", type=int, default=None, help="causal chunks per head"
    )
    parser.add_argument("--approx", action="store_true", help="approximate exponential")
    parser.add_argument(
        "--noc", type=int, choices=[0, 1], default=0, help="broadcast's NoC"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--cards", default="", help="e.g. 0,1: a session of islands, every flat rank"
    )
    parser.add_argument(
        "--far", action="store_true", help="with --cards: O held by island 1 alone"
    )
    parser.add_argument("--save", default=None, help="O's raw pages to this .npy")
    return parser.parse_args()


def shape_dims(name):
    """(b, h, hk, s, d, causal) of a preset and its Dims (Sq = Sk, D = Dv)."""
    batches, heads, kv_heads, sequence_length, head_dim, causal = pl.SHAPES[name]
    return (batches, heads, kv_heads, sequence_length, head_dim, causal), pl.dims(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        causal,
    )


def check(
    session, output, queries, keys, values, call, causal, approx=False, count=SAMPLED
):
    """Sampled query tile rows of O against float64: error statistics and `ok`."""
    group = call.h // call.hk
    got, want = [], []
    for batch, head, row in sdpa.sample_rows(call.b, call.h, call.sqt, count):
        first = ((batch * call.h + head) * call.sqt + row) * call.dvt
        tiles = session.read_tiles_bf16(output, first, call.dvt)
        got.append(np.concatenate(list(tiles), axis=1))
        rows = np.arange(row * pl.TILE, (row + 1) * pl.TILE)
        kv_head = head // group
        want.append(
            sdpa.attention(
                queries[batch, head],
                keys[batch, kv_head],
                values[batch, kv_head],
                rows,
                causal,
            )
        )
    statistics = sdpa.errors(np.concatenate(got), np.concatenate(want))
    statistics["tile_rows"] = len(got)
    rms_bound = APPROX_RMS_BOUND if approx else RMS_BOUND
    statistics["ok"] = (
        statistics["rms_rel"] <= rms_bound and statistics["max_rel"] <= MAX_BOUND
    )
    return statistics


def main():
    arguments = parse()
    (batches, heads, kv_heads, sequence_length, head_dim, causal), call = shape_dims(
        arguments.shape
    )
    queries, keys, values = sdpa.inputs(
        batches,
        heads,
        kv_heads,
        sequence_length,
        sequence_length,
        head_dim,
        head_dim,
        arguments.seed,
    )
    scale = 1.0 / np.sqrt(head_dim)
    report = {
        "shape": arguments.shape,
        "b": batches,
        "h": heads,
        "hk": kv_heads,
        "s": sequence_length,
        "d": head_dim,
    }
    report["causal"], report["approx"], report["noc"] = (
        causal,
        arguments.approx,
        arguments.noc,
    )
    cards = [int(n) for n in arguments.cards.split(",")] if arguments.cards else []
    opened = {"cards": cards, "links": 1} if len(cards) > 1 else {}
    report["cards"], report["far"] = cards, arguments.far
    with km.Session(log=log, **opened) as session:
        grid, held = None, {}
        if opened:  # one grid of the islands, every flat rank
            plan, grid = card.island_plan(
                session, call, length=arguments.broadcast, chunks=arguments.chunks
            )
            if arguments.far:
                graph = session.graph
                held = {"split": 0, "group": (graph.first(1), graph.count(1))}
        else:
            nocs = session.nocs()
            plan = pl.make(
                call, nocs.grid, length=arguments.broadcast, chunks=arguments.chunks
            )
        report["plan"] = plan.summary()
        started = time.perf_counter()
        zeros = np.zeros((batches, heads, sequence_length, head_dim), np.float32)
        tensors = {
            "q": card.bf16_tensor(session, queries),
            "k": card.bf16_tensor(session, keys),
            "v": card.bf16_tensor(session, values),
            "o": card.bf16_tensor(session, zeros, **held),
            "tables": card.table_tensor(session, pl.tables(plan)),
        }
        report["upload_s"] = round(time.perf_counter() - started, 3)
        log(f"uploaded, plan {report['plan']}")
        attention = card.Attention(
            session,
            plan,
            tensors,
            scale,
            arguments.approx,
            noc=arguments.noc,
            grid=grid,
        )
        log("loaded")
        host_s, command_microseconds = attention.calls(arguments.calls)
        log(f"{arguments.calls} calls in {host_s:.4f} s")
        report["run"] = attention.stop()
        log("stopped")
        report["check"] = check(
            session, tensors["o"], queries, keys, values, call, causal, arguments.approx
        )
        log("sampled rows read back")
        if arguments.save:  # every page of O, for a comparison on the host
            (pages,) = session.read(tensors["o"])
            np.save(arguments.save, pages)
            log(f"O saved: {arguments.save}")
    report["ok"] = report["check"]["ok"] and report["run"]["ok"]
    print(json.dumps(power.clocked(report)), flush=True)
    if arguments.json:
        Path(arguments.json).write_text(json.dumps(report))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
