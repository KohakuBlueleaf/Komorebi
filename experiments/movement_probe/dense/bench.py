"""The matrix product with one stream split over both movers by a pattern (resident
form: Y's column blocks; --split: the K chunks), against the program's own kernels:
single calls, a loop, the check of Y's sampled rows, a profile or the parts-only cut.
Example: python experiments/movement_probe/dense/bench.py --shape lin384 main BBBA"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import komorebi as km  # noqa: E402
from komorebi import abi  # noqa: E402
from komorebi.session import aiclk_mhz, visible_card  # noqa: E402
from projects.dense import bench, card, run  # noqa: E402
from projects.dense import plan as pl  # noqa: E402
from reference import dense as ref  # noqa: E402

KERNELS = Path(__file__).resolve().parent / "kernels"
HEADER = KERNELS / "dense.h"
SPLIT_KERNELS = KERNELS / "split"
K = abi.constants(HEADER, known=abi.headers().constants)
S = abi.constants(SPLIT_KERNELS / "split.h", known=abi.headers().constants)
log = run.log


def movement_program(p, a_tensor, pattern, fidelity="HiFi2"):
    """Plan p's program with the movement kernels: Y's blocks owned by `pattern`."""
    arguments = a_tensor.accessor_args()
    buffers = [km.Buffer(i, "bfloat16", pl.TILE_BYTES, n) for i, n in p.buffers()]
    buffers += [
        km.Buffer(K["kOutputMoverA"], "bfloat16", pl.TILE_BYTES, p.out_pages),
        km.Buffer(K["kCbProf"], "uint32", K["kProfBytes"], 1),
        km.Buffer(K["kCbLoaded"], "uint32", K["kLoadedBytes"], 1),
    ]
    movement = KERNELS / "movement.cpp"
    return km.Program(
        "dense_movement",
        kernels={
            "reader": movement,
            "writer": movement,
            "compute": KERNELS / "compute.cpp",
        },
        buffers=buffers,
        compile_args={"reader": arguments, "writer": arguments},
        compute={"math_fidelity": fidelity, "fp32_dest_acc_en": True},
        defines={"Y_PATTERN": km.Pattern(pattern).word},
        handlers=HEADER,
    )


class Product(card.Product):
    """Plan p loaded with W resident: the program's own kernels (pattern None) or the
    movement kernels with `pattern`; LOAD at the start."""

    def __init__(self, session, p, tensors, count, pattern, deadline):
        self.p, self.t, self.group = p, tensors, (0, count)
        made = (
            card.program(p, tensors["a"])
            if pattern is None
            else movement_program(p, tensors["a"], pattern)
        )
        self.run = session.load(made, deadline=deadline, wait=1.0)
        handle = self.run.submit(
            "load", self.group, pl.load_args(p, tensors["w"].address)
        )
        self.load_microseconds = self.run.wait(handle, timeout=10.0).microseconds

    def single(self, n, mode=K["kModeFull"]):
        out = []
        for _ in range(n):
            handle = self.run.submit("product", self.group, self.words(mode))
            out.append(self.run.wait(handle, timeout=10.0).microseconds)
        return np.array(out)


def split_program(sp, tensors, pattern):
    """K-split plan sp's program with the chunks owned by `pattern`: main's two-chunk
    buffers become two one-chunk halves (same L1); the writer gets A's and W's tensors.
    """
    made = card.split_program(sp, tensors)
    halves = {S["kCbSa"]: S["kChunkAMoverB"], S["kCbSw"]: S["kChunkWMoverB"]}
    split = [b for b in made.buffers if b.index in halves]
    made.buffers = [b for b in made.buffers if b.index not in halves]
    for b in split:
        made.buffers += [
            km.Buffer(index, b.format, b.page_bytes, b.pages // 2)
            for index in (b.index, halves[b.index])
        ]
    made.kernels = {
        k: SPLIT_KERNELS / f"{k}.cpp" for k in ("reader", "writer", "compute")
    }
    made.compile_args["writer"] = (
        tensors["y"].accessor_args()
        + tensors["a"].accessor_args()
        + tensors["w"].accessor_args()
    )
    made.defines = {"CHUNK_PATTERN": km.Pattern(pattern).word}
    return made


class SplitProduct(card.SplitProduct):
    """K-split plan sp loaded with the program's own kernels (pattern None) or with its
    chunks split by `pattern`."""

    def __init__(self, session, sp, tensors, reduce, pattern, deadline):
        self.g, self.t, self.form = sp, tensors, reduce
        self.group = (0, sp.cores())
        made = (
            card.split_program(sp, tensors)
            if pattern is None
            else split_program(sp, tensors, pattern)
        )
        self.run = session.load(made, deadline=deadline, wait=1.0)
        self.load_microseconds = None

    def single(self, n, mode=S["kSplitFull"]):
        out = []
        for _ in range(n):
            handle = self.run.submit(
                self.handler, self.group, self.words(mode), exclusive=True
            )
            out.append(self.run.wait(handle, timeout=10.0).microseconds)
        return np.array(out)


def make_product(session, p, tensors, count, pattern, args):
    deadline = bench.deadline(p)
    if isinstance(p, pl.Split):
        return SplitProduct(session, p, tensors, args.reduce, pattern, deadline)
    return Product(session, p, tensors, count, pattern, deadline)


def measure(session, p, tensors, count, pattern, args, x, w):
    """One layout: single calls, the loop, the check, then one profiled call."""
    index = visible_card()
    out = {"layout": pattern or "main"}
    tensors["y"].write(np.zeros((p.mt * pl.TILE, p.nt * pl.TILE), np.float32))
    log(f"{out['layout']}: Y cleared")
    product = make_product(session, p, tensors, count, pattern, args)
    single = product.single(21)
    one = float(np.median(single))
    out["single_microseconds"] = {"median": one, "min": float(single.min())}
    log(f"{out['layout']}: single call {one:.2f} us")
    reps = max(2, math.ceil(args.seconds * 1e6 / one))
    out["loop"] = bench.loop(product, reps, index, one, reps)
    log(f"{out['layout']}: loop {out['loop']['microseconds_per_call']} us per call")
    split = isinstance(p, pl.Split)
    if split:  # the parts only: the K chunks read and multiplied, nothing summed
        parts = max(2, math.ceil(1e6 / one))
        host_s, _ = product.calls(parts, mode=S["kSplitParts"])
        out["parts_only_microseconds_per_call"] = round(host_s * 1e6 / parts, 3)
    product.calls(1)
    out["run"] = product.stop()
    rows = ref.sampled_rows(p.mt * pl.TILE)
    got = card.read_rows(session, tensors["y"], rows, log)
    out["check"] = run.check(got[:, : p.nt * pl.TILE], rows, x, w, "uniform")
    if not split:
        product = make_product(session, p, tensors, count, pattern, args)
        product.single(1)
        prof = product.profile(session, tensors["profile"])
        out["profile"] = bench.profile_summary(prof, aiclk_mhz(index), False)
        product.stop()
    out["ok"] = out["check"]["ok"] and run.clean(out["run"])
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("patterns", nargs="+", help="main, or a pattern of A and B")
    a.add_argument("--shape", choices=run.SHAPES, required=True)
    a.add_argument("--split", default=None, help="a K-split plan s,oh,ow[,kc]")
    a.add_argument("--reduce", choices=list(pl.REDUCE_FORMS), default="owner")
    a.add_argument("--seconds", type=float, default=2.0)
    a.add_argument("--cool", type=float, default=10.0, help="s between layouts")
    a.add_argument("--json", type=Path, help="append the JSON lines to this file")
    args = a.parse_args()
    m, k, n = run.SHAPES[args.shape]
    form = "split" if args.split else "resident"
    p = run.make_plan(form, (m, k, n), None, args.split)
    log("building the inputs")
    x, w = ref.inputs(m, k, n)
    with km.Session(log=log) as session:
        cores = len(session.nocs().words)
        count = p.cores() if args.split else pl.group_size(p, cores)
        tensors = {"profile": card.prof_tensor(session, cores)}
        for name, value in (("a", x), ("w", w), ("y", np.zeros((m, n), np.float32))):
            tensors[name] = card.bf16_tensor(session, value)
            log(f"{name.upper()} uploaded")
        for i, pattern in enumerate(args.patterns):
            if i:
                for _ in range(int(args.cool // 5)):
                    log("cooling")
                    time.sleep(5)
            line = {"shape": args.shape, "plan": run.describe(p)}
            line |= measure(
                session,
                p,
                tensors,
                count,
                None if pattern == "main" else pattern,
                args,
                x,
                w,
            )
            line["load_avg"] = Path("/proc/loadavg").read_text().split()[:3]
            print(json.dumps(line, default=float), flush=True)
            if args.json:
                with args.json.open("a") as f:
                    f.write(json.dumps(line, default=float) + "\n")


if __name__ == "__main__":
    main()
