"""The card's two ceilings for the dense project's bounds (kernels/ceilings): READ,
every rank reading its share of a tensor from the banks on both movers (nothing
kept): GB/s of the card; MATMUL, the matrix unit of every rank multiplying resident
tiles at the linear kernels' fidelity (nothing read): tile products a second. Prints
one JSON line."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "kernels" / "ceilings"
HEADER = KERNELS / "ceilings.h"
C = abi.constants(HEADER, known=abi.headers().constants)
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def calls(run, handler, group, words, commands):
    """`commands` commands back to back: (host seconds, each command's slowest part
    in microseconds)."""
    table = np.tile(np.array(words, np.int64), (commands, 1))
    started = time.perf_counter()
    handles = run.submit_many(handler, *group, table, exclusive=True)
    events = run.wait(list(handles), timeout=60.0)
    host_s = time.perf_counter() - started
    bad = [part for event in events for part in event if part.kind != "done"]
    if bad:
        raise km.KomorebiError(f"{len(bad)} parts not done: {bad[0]}")
    return host_s, [max(part.microseconds for part in event) for event in events]


def measure(run, handler, group, words, singles=3, loop=20):
    """Single calls one at a time, then `loop` back to back: microseconds."""
    single = [calls(run, handler, group, words, 1)[1][0] for _ in range(singles)]
    host_s, parts = calls(run, handler, group, words, loop)
    return {
        "single_us": [round(value, 1) for value in single],
        "loop_us_per_call": round(host_s * 1e6 / loop, 1),
        "loop_part_us_median": round(float(np.median(parts)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--mib", type=int, default=128, help="the tensor read, MiB")
    parser.add_argument("--chunks", default="4,8,16", help="pages a chunk, each run")
    parser.add_argument("--passes", type=int, default=2000, help="MATMUL passes")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    tile = C["kTileBytes"]
    pages = args.mib * (1 << 20) // tile
    report = {"card": args.card, "mib": args.mib, "pages": pages}
    blocks = C["kBm"] * C["kKb"] + C["kKb"] * C["kBn"]
    with km.Session(cards=[args.card], log=log) as session:
        one = session.sessions[0] if hasattr(session, "sessions") else session
        tensor = session.empty((pages * 32, 32), "bfloat16", "tile")
        program = km.Program(
            "dense_ceilings",
            kernels=KERNELS,
            buffers=[
                km.Buffer(C["kCbRead"], "bfloat16", tile, C["kChunkPages"]),
                km.Buffer(C["kCbWrite"], "bfloat16", tile, C["kChunkPages"]),
                km.Buffer(C["kCbA"], "bfloat16", tile, C["kBm"] * C["kKb"]),
                km.Buffer(C["kCbB"], "bfloat16", tile, C["kKb"] * C["kBn"]),
                km.Buffer(C["kCbOut"], "float32", 4096, 1),
            ],
            compile_args={
                "reader": tensor.accessor_args(),
                "writer": tensor.accessor_args(),
                "compute": [],
            },
            compute={"math_fidelity": "HiFi2", "fp32_dest_acc_en": True},
            handlers=HEADER,
        )
        run = session.load(program, wait=5.0)
        group = (0, session.ranks)
        report["ranks"] = session.ranks
        try:
            report["banks"] = int(one._module.dram_banks(one.device))
        except (AttributeError, TypeError):
            report["banks"] = 8  # a p150a's GDDR6 banks
        report["read"] = {}
        for chunk in (int(value) for value in args.chunks.split(",")):
            row = measure(run, "read", group, [tensor.address, pages, chunk])
            best = min(row["single_us"])
            row["gb_per_s_single_best"] = round(pages * tile / best / 1e3, 1)
            row["gb_per_s_loop"] = round(
                pages * tile / row["loop_us_per_call"] / 1e3, 1
            )
            row["gb_per_s_a_bank"] = round(
                row["gb_per_s_single_best"] / report["banks"], 1
            )
            report["read"][chunk] = row
            log(f"read, chunks of {chunk}: {row}")
        row = measure(run, "matmul", group, [args.passes])
        products = args.passes * C["kBm"] * C["kBn"] * C["kKb"] * session.ranks
        best = min(row["single_us"])
        row["tile_products"] = products
        row["tile_products_per_s_single"] = round(products / best * 1e6)
        row["cycles_a_tile_product_at_1350"] = round(
            best * 1350 / (products / session.ranks), 2
        )
        report["matmul"] = row
        report["resident_tiles"] = blocks
        log(f"matmul: {row}")
        run.stop()
    report["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
    print(json.dumps(report), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
