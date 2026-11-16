"""Y = X W (32 x 1024 x 1024) on 8 cores, the weight in bf16, bfp8_b or bfp4_b, at
LoFi, HiFi2 and HiFi4: with no marker (X in SrcB) and with the weight marked km.srcb
(turned), each against float64 (README.md). `build` builds each program on a mock
device and prints the build's placement lines; `card` runs them on one card."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import komorebi as km
from komorebi import formats
from komorebi.device import native
from komorebi.session import process
from komorebi.session import session as km_session

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
JIT_CACHE = ROOT / "build" / "ttcache-formats"  # probe.py's: the jobs read it
DESCRIPTOR = (
    "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)
K_TILES = 32  # K = 1024
N_TILES = 32  # N = 1024
WIDTH = 4  # Y's columns of tiles a core: 8 cores
CORES = N_TILES // WIDTH
CHUNK = 2  # K tiles a block
WEIGHTS = {"16": "bfloat16", "8": "bfp8_b", "4": "bfp4_b"}
FIDELITIES = ("LoFi", "HiFi2", "HiFi4")
REPEAT = 3
STARTED = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def proof_class(fidelity):
    """The class at `fidelity`: for each weight format a handler unmarked (wN) and one
    marked (tN, the tensor of transposed tiles), each core WIDTH columns of Y."""

    class Proof(km.Program):
        sums: km.L1[km.fp32, 1, WIDTH, 32, 32]

        @km.tile_function(fidelity=fidelity)
        def plain(x, w):
            return x @ w

        @km.tile_function(fidelity=fidelity)
        def turned(x, w):
            return x @ km.srcb(w)

        @km.handler
        def plain16(self, x: km.Tensor, w16: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(w16.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.plain(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

        @km.handler
        def turned16(self, x: km.Tensor, t16: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(t16.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.turned(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

        @km.handler
        def plain8(self, x: km.Tensor, w8: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(w8.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.plain(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

        @km.handler
        def turned8(self, x: km.Tensor, t8: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(t8.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.turned(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

        @km.handler
        def plain4(self, x: km.Tensor, w4: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(w4.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.plain(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

        @km.handler
        def turned4(self, x: km.Tensor, t4: km.Tensor, y: km.Tensor):
            first = self.rank * WIDTH
            for xb, wb in zip(
                km.Stream(x.tiles[0:1, 0:K_TILES], block=CHUNK),
                km.Stream(t4.tiles[0:K_TILES, first : first + WIDTH], block=8),
            ):
                self.sums += self.turned(xb, wb)
            y.tiles[0:1, first : first + WIDTH].write(self.sums)

    Proof.__name__ = Proof.__qualname__ = f"Proof{fidelity}"
    return Proof


def inputs(seed=1):
    """X (bfloat16 values) and W (float32, normal / 32)."""
    rng = np.random.default_rng(seed)
    bits = rng.standard_normal((32, K_TILES * 32)).astype(np.float32).view(np.uint32)
    x = ((bits + 0x8000) & 0xFFFF0000).view(np.float32)
    w = (rng.standard_normal((K_TILES * 32, N_TILES * 32)) / 32).astype(np.float32)
    return x, w


def tensors_of(session, x, w, empty):
    """The tensors by parameter: X, each weight plain (wN) and of transposed tiles
    (tN), Y (float32); `empty` (the build): not written."""
    out = {}
    for key, dtype in WEIGHTS.items():
        if empty:
            out[f"w{key}"] = session.empty(w.shape, dtype, "tile")
            out[f"t{key}"] = session.empty(w.shape, dtype, "tile")
            out[f"t{key}"].transposed_tiles = True
        else:
            out[f"w{key}"] = session.tensor(w, dtype, "tile")
            out[f"t{key}"] = session.tensor(w, dtype, "tile", transposed_tiles=True)
    if empty:
        out["x"] = session.empty(x.shape, "bfloat16", "tile")
    else:
        out["x"] = session.tensor(x, "bfloat16", "tile")
    out["y"] = session.empty((32, N_TILES * 32), "float32", "tile")
    return out


def build_only():
    """Session.load cut down to the runtime's kernel build (the sizes test's child):
    returns its sizes row instead of stopping the script."""
    path = ROOT / "tests" / "projects" / "kernel_sizes_child.py"
    spec = importlib.util.spec_from_file_location("kernel_sizes_child", path)
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)

    def load(self, program, **kwargs):
        try:
            child.build_only(self, program, **kwargs)
        except child.Built:
            return None
        raise RuntimeError("no build")

    return load


def build(args):
    """Mock device: each fidelity's class made and built; its placement lines."""
    tree = native.metal_tree()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(tree / DESCRIPTOR)
    process.chip_holder = lambda card, path=None: None  # a mock device has no chip
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    km_session.Session.load = build_only()
    x, w = inputs()
    with km.Session(log=log) as session:
        tensors = tensors_of(session, x, w, empty=True)
        for fidelity in FIDELITIES:
            program = proof_class(fidelity)(log=log, session=session, **tensors)
            session.load(program)
    return 0


def errors(got, want):
    """The relative RMS error and the largest error over Y's RMS."""
    scale = np.sqrt((want**2).mean())
    return {
        "rms_rel": float(np.sqrt(((got - want) ** 2).mean()) / scale),
        "max_over_rms": float(np.abs(got - want).max() / scale),
    }


def card(args):
    """One card: each fidelity's class loaded in turn, each handler REPEAT commands on
    CORES cores; Y read after the run; the table's rows."""
    os.environ["TT_METAL_CACHE"] = str(JIT_CACHE)
    x, w = inputs()
    want = x.astype(np.float64) @ w.astype(np.float64)
    rows = []
    with km.Session(log=log) as session:
        tensors = tensors_of(session, x, w, empty=False)
        for fidelity in FIDELITIES:
            cls = proof_class(fidelity)
            program = cls(log=log, session=session, **tensors)
            run = session.load(program)
            outs = {}
            try:
                for key in WEIGHTS:
                    for form in ("plain", "turned"):
                        handler = getattr(cls, f"{form}{key}")
                        y = session.empty((32, N_TILES * 32), "float32", "tile")
                        weight = tensors[("t" if form == "turned" else "w") + key]
                        times = []
                        for _ in range(REPEAT):
                            sent = run.submit(
                                handler, (0, CORES), args=(tensors["x"], weight, y)
                            )
                            event = run.wait(sent, timeout=30.0)
                            if event.kind != "done":
                                raise RuntimeError(f"{handler.name}: {event}")
                            times.append(event.microseconds)
                        outs[(key, form)] = (y, times)
            finally:
                run.stop()
            for (key, form), (y, times) in outs.items():
                dtype = WEIGHTS[key]
                row = {"fidelity": fidelity, "weight": dtype}
                row["form"] = "no marker" if form == "plain" else "w marked km.srcb"
                row |= errors(y.read().astype(np.float64), want)
                row["microseconds"] = times
                row["weight_bytes"] = (K_TILES * N_TILES) * formats.tile_bytes(dtype)
                stored = formats.from_bytes(
                    *formats.to_bytes(w, dtype, "tile", form == "turned"),
                    dtype,
                    "tile",
                    form == "turned",
                )
                row["w_stored_rms_rel"] = errors(stored, w)["rms_rel"]
                rows.append(row)
                log(json.dumps(row))
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "placement.json", "w") as f:
        json.dump({"rows": rows, "wall_s": round(time.monotonic() - STARTED, 1)}, f)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["build", "card"])
    args = p.parse_args()
    return build(args) if args.what == "build" else card(args)


if __name__ == "__main__":
    sys.exit(main())
