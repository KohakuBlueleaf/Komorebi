"""The shared load on a session's islands (README.md, "On more islands"): the four
pieces' classes in one load; its kernels' bytes, its phases, a switch on the first
island and on all, and a host write on every island after a switch."""

import argparse
import time

import numpy as np
from galaxy.driver.interface import load_factory, load_weights
from galaxy.driver.run import open_session, parse_layers
from galaxy.reference.model_loader import Checkpoint

import komorebi as km
from komorebi import formats
from projects.galaxy.load.shared import Shared

PIECES = (  # a factory and the weights it takes ("W": the loader's, "ckpt", None)
    ("galaxy.experts.piece:make", "W"),
    ("galaxy.attention_local.piece:make", None),
    ("galaxy.parts.piece:make", None),
    ("galaxy.attention_global.piece:make_compress_index", "ckpt"),
)


def start(shared, session):
    """Shared.start's steps, each timed: the dict's load, then each added class."""
    t0 = time.monotonic()
    first = next(iter(shared.groups))
    rest = [p for p in shared.groups if p is not first]
    for program in rest:
        first.ahead(program)
    first.slots(count=1, service="shared")
    shared.first = first
    shared.run = session.load({first: (0, session.ranks)}, reserve=shared.reserve)
    t1 = time.monotonic()
    added = []
    for program in rest:
        ta = time.monotonic()
        shared.run.addon(program)
        added.append((program.name, round(time.monotonic() - ta, 2)))
    shared.held = [first] * session.ranks
    print(f"load of {first.name}: {t1 - t0:.1f} s; added {added}", flush=True)
    print(f"start: {time.monotonic() - t0:.1f} s, {len(shared.groups)} programs")


def sizes(shared, session):
    """The load's kernels, the bytes the kernel buffer holds: the dict's program's
    build (the added classes' code is in their parts, in L1 slots)."""
    built = shared.first.sizes
    print(
        f"kernel_bytes {session.kernel_bytes:,} B; the load's kernels {built.total:,} B"
    )
    print(f"{shared.first.name}: text {built.text}")


def switch_ms(shared, program, group, times=5):
    """A switch's host time (ms), each from the dict's program back to it."""
    out = []
    for _ in range(times):
        t0 = time.perf_counter()
        event = shared.run.wait(shared.load(program, group), timeout=60.0)
        out.append(round((time.perf_counter() - t0) * 1e3, 2))
        if event.kind != "done":
            raise km.KomorebiError(f"switch to {program.name} on {group}: {event}")
        shared.run.wait(shared.load(shared.first, group), timeout=60.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kernel-bytes", type=int, default=104_448)
    a = ap.parse_args()
    session = open_session(km, parse_layers(a.cards, 0), a.kernel_bytes)
    ranks, first_island = session.ranks, session.graph.count(0)
    print(
        f"{len(session.graph.islands)} islands, {ranks} ranks, island 0 {first_island}"
    )
    weights = {
        "W": load_weights(session, "2,3@ffn.", parse_layers),
        "ckpt": Checkpoint(a.ckpt),
        None: None,
    }
    shared = Shared(session)
    for spec, kind in PIECES:
        load_factory(spec)(session, weights[kind], [2, 3], run=shared)
    start(shared, session)
    sizes(shared, session)
    named = {type(p).__name__: p for p in shared.groups}
    for name in ("Window", "Projections", "RmsNorm", "Moe"):
        one = switch_ms(shared, named[name], (0, first_island))
        every = switch_ms(shared, named[name], (0, ranks))
        print(f"switch {name}: island 0 {one} ms; every island {every} ms")
    shared.run.wait(shared.load(named["Window"], (0, ranks)), timeout=60.0)
    x = session.empty((32 * ranks, 32), "bfloat16", "tile")  # a tile on every rank
    values = np.random.default_rng(0).standard_normal((32 * ranks, 32))
    raw, _ = formats.to_bytes(values.astype(np.float32), "bfloat16", "tile")
    raw = np.asarray(raw).view(np.uint8)
    session.write(x, raw.reshape(x.pages, -1))
    (back,) = session.read(x, first=0, pages=x.pages)
    same = np.array_equal(np.asarray(back).view(np.uint8).ravel(), raw.ravel())
    print(f"a write on every island after a switch: {'equal' if same else 'DIFFERS'}")
    shared.run.stop()


if __name__ == "__main__":
    main()
