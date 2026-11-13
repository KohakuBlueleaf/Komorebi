"""Running the matrix product on a card through `komorebi.Session`: the program of a
plan, the operand tensors, and the commands (LOAD, PRODUCT, PROF) submitted through the
dispatcher on X280 cluster 0. Needs tt-metal's Python environment.
"""

import time

import numpy as np

import komorebi as km
from komorebi import _env, abi, layout
from projects.dense import plan as pl
from ref import dense as ref

K, G, S = pl.K, pl.G, pl.S
KERNELS = pl.HERE / "kernels"
GRID_KERNELS = KERNELS / "grid"
SPLIT_KERNELS = KERNELS / "split"
BROADCAST_FIELDS = abi.struct_fields(
    _env.include_dir() / "komorebi" / "device" / "km_broadcast.hpp", "BroadcastArgs"
)


def bf16_tensor(session, x):
    """A float array (bf16 values) as a bf16 tile tensor in DRAM."""
    return session.tensor(np.asarray(x, dtype=np.float32), "bfloat16", "tile")


def read_rows(session, y, rows, log=None):
    """Rows `rows` of the bf16 tile tensor Y as float32, read one row tile at a time
    (only the tiles holding them, at most 1 MB each), after the run's stop."""
    nc = y.shape[1] // ref.TILE
    tiles = np.unique(np.asarray(rows) // ref.TILE)
    got = {}
    for i, t in enumerate(tiles):
        part = session.read_tiles_bf16(y, int(t) * nc, nc)  # [nc, 32, 32]
        got[int(t)] = part.transpose(1, 0, 2).reshape(ref.TILE, nc * ref.TILE)
        if log:
            log(f"row tile {i + 1} of {len(tiles)} read")
    return np.stack([got[r // ref.TILE][r % ref.TILE] for r in rows])


def prof_tensor(session, cores):
    """Zeroed profile pages in DRAM, one page of kProfBytes per core (row-major)."""
    return session.tensor(np.zeros((cores, K["kProfBytes"] // 4), np.int32), "int32")


def program(p, a_tensor, fidelity="HiFi2"):
    """The persistent program of plan p: the three kernels and the plan's buffers."""
    ct = a_tensor.accessor_args()
    buffers = [km.Buffer(i, "bfloat16", pl.TILE_BYTES, n) for i, n in p.buffers()]
    buffers += [
        km.Buffer(K["kCbProf"], "uint32", K["kProfBytes"], 1),
        km.Buffer(K["kCbLoaded"], "uint32", K["kLoadedBytes"], 1),
    ]
    return km.Program(
        "dense",
        kernels=KERNELS,
        buffers=buffers,
        compile_args={"reader": ct, "writer": ct},
        compute={"math_fidelity": fidelity, "fp32_dest_acc_en": True},
        handlers=pl.DENSE_H,
    )


class Product:
    """Plan p loaded on a session's grid with operand tensors a, w, y: LOAD at the
    start (resident form), then PRODUCT calls and PROF on the group (0, count)."""

    def __init__(self, session, p, tensors, count, fidelity="HiFi2", deadline=2.0):
        self.p, self.t, self.group = p, tensors, (0, count)
        self.run = session.load(
            program(p, tensors["a"], fidelity), deadline=deadline, wait=1.0
        )
        self.load_us = None
        if p.resident:
            w = tensors["w"].address
            handle = self.run.submit("load", self.group, pl.load_args(p, w))
            self.load_us = self.run.wait(handle, timeout=10.0).us

    def words(self, mode):
        t = self.t
        return pl.product_args(
            self.p,
            t["a"].address,
            t["y"].address,
            t["w"].address,
            mode,
        )

    def calls(self, n, mode=K["kModeFull"], timeout=60.0):
        """n PRODUCT commands back to back: (host seconds from the first submit to the
        last event, each command's submit-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (n, 1))
        t0 = time.perf_counter()
        handles = self.run.submit_many("product", self.group[0], self.group[1], words)
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} commands not done: {bad[0]}")
        return host_s, np.array([e.us for e in events])

    def single(self, n, mode=K["kModeFull"]):
        """n PRODUCT commands one at a time (each waited for): microseconds each."""
        out = []
        for _ in range(n):
            handle = self.run.submit("product", self.group, self.words(mode))
            out.append(self.run.wait(handle, timeout=10.0).us)
        return np.array(out)

    def profile(self, session, prof):
        """Each core's profile page (summed over the commands so far): [cores,
        words]."""
        handle = self.run.submit("prof", self.group, [prof.address])
        self.run.wait(handle, timeout=10.0)
        (pages,) = session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : K["kProfWords"]]

    def stop(self):
        """Stops the run; the kernel's result and the cores that gave up."""
        self.run.stop()
        return self.summary()

    def failed(self, err):
        """After a failed wait: the run stopped if it can be, and what its state shows
        (each read guarded, for the log before the driver leaves)."""
        out = {"error": f"{type(err).__name__}: {err}"}
        try:
            self.run.stop()
        except km.KomorebiError as stop_err:
            out["stop"] = f"{type(stop_err).__name__}: {stop_err}"
        try:
            out["broadcasts"] = broadcasts_at_stall(self.run.state())
            out |= self.summary()
        except km.KomorebiError as state_err:
            out["state"] = f"{type(state_err).__name__}: {state_err}"
        return out

    def summary(self):
        """The stopped run's kernel result, cores ended, give-ups and broadcast
        waits."""
        states = self.run.state()
        return {
            "cores": len(states),
            "kernel": (self.run.result() or {}).get("error_name"),
            "cores_ended": {
                k: sum(c[k] != 0 for c in states)
                for k in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {k: c[k] for k in ("core", "reader", "writer", "compute")}
                for c in states
                if c["reader"] or c["writer"] or c["compute"]
            ][:8],
            "broadcast_wait_cycles_max": [
                max(c["broadcasts"][broadcast]["wait_cycles"] for c in states)
                for broadcast in (0, 1)
            ],
            "broadcast_gave_up": [
                {"core": c["core"]} | area["gave_up"]
                for c in states
                for area in c["broadcasts"]
                if area["gave_up"]
            ][:8],
        }


def broadcasts_at_stall(states):
    """By broadcast: blocks delivered (min, max), the cores at the min (first 8),
    their VALID words, and the cores' commands completed (min, max)."""
    out = {
        "completed": [
            min(c["completed"] for c in states),
            max(c["completed"] for c in states),
        ]
    }
    for broadcast in (0, 1):
        blocks = [c["broadcasts"][broadcast]["blocks"] for c in states]
        low = [c for c in states if c["broadcasts"][broadcast]["blocks"] == min(blocks)]
        out[f"broadcast{broadcast}"] = {
            "blocks": [min(blocks), max(blocks)],
            "at_min": [c["core"] for c in low][:8],
            "valid_at_min": [c["broadcasts"][broadcast]["valid"] for c in low][:8],
        }
    return out


def grid_broadcasts(g, nocs, group, along, noc, buffer):
    """Broadcast words by core for plan g: broadcasts along the rect's rows or
    columns, or a broadcast of one core each where the layout gives every core its own
    operand."""
    common = (nocs.grid, nocs.panels, group)
    at = (nocs.noc0, nocs.noc_xy)
    rect = g.rect(nocs.grid)
    out = layout.broadcast_args(*common, rect, along, noc, *at, buffer)
    if {"row": "rows", "column": "cols"}[along] != g.layout:
        return out
    gx = nocs.grid[0]
    for core in g.places(nocs.grid):
        y, x = divmod(core, gx)
        one = layout.broadcast_args(*common, (x, y, 1, 1), along, noc, *at, buffer)
        out[core] = one[core]
    return out


def grid_program(g, nocs, tensors, fidelity="HiFi2"):
    """The grid form's program for plan g on a grid whose NoCs are `nocs`: each core's
    broadcast words (row broadcast on NoC0, column broadcast on NoC1), place and plan
    words."""
    group = layout.Group(*g.group(nocs.grid))
    row = grid_broadcasts(
        g, nocs, group, "row", 0, (G["kBroadcastRow"], G["kCbRow"], g.wa, g.bm * g.kb)
    )
    col = grid_broadcasts(
        g,
        nocs,
        group,
        "column",
        1,
        (G["kBroadcastCol"], G["kCbCol"], g.ww, g.kb * g.bn),
    )
    places = g.places(nocs.grid)
    idle = [0] * (len(BROADCAST_FIELDS) + G["kCoreWords"])

    def args(core):
        if core not in group.indices:
            return idle, idle, idle[: G["kCoreWords"]] + [g.wa, g.ww]
        m0, n0 = places.get(core, (0, 0))
        mine = [int(core in places), m0, n0, *g.words()]
        broadcasts = [
            [words[core][f] for f in BROADCAST_FIELDS] for words in (row, col)
        ]
        return broadcasts[0] + mine, broadcasts[1] + mine, mine + [g.wa, g.ww]

    return km.Program(
        "dense_grid",
        kernels=GRID_KERNELS,
        buffers=[km.Buffer(*b) for b in g.buffers()],
        args=args,
        compile_args={
            "reader": tensors["a"].accessor_args(),
            "writer": tensors["w"].accessor_args() + tensors["y"].accessor_args(),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {G["kCbPart"]: "UnpackToDestFp32"},
        },
        handlers=pl.GRID_H,
    )


class GridProduct:
    """Plan g (the grid form) loaded on a session with operand tensors a, w, y: one
    GRID command per call on the rectangle's group, exclusive (its broadcasts)."""

    handler = "grid"
    modes = {
        "full": G["kGridFull"],
        "broadcasts": G["kGridBroadcasts"],
        "product": G["kGridProduct"],
    }

    def __init__(self, session, g, tensors, fidelity="HiFi2", deadline=2.0, wait=1.0):
        self.g, self.t = g, tensors
        nocs = session.nocs()
        self.group = g.group(nocs.grid)
        program = grid_program(g, nocs, tensors, fidelity)
        self.run = session.load(program, deadline=deadline, wait=wait)
        self.load_us = None

    def words(self, mode):
        t = self.t
        addr = [t[k].address for k in ("a", "w", "y")]
        return pl.grid_args(self.g, *addr, mode)

    def calls(self, n, mode=G["kGridFull"], timeout=60.0):
        """n GRID commands back to back: (host seconds from the first submit to the last
        event, each command's submit-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (n, 1))
        t0 = time.perf_counter()
        first, count = self.group
        handles = self.run.submit_many(
            self.handler, first, count, words, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} commands not done: {bad[0]}")
        return host_s, np.array([e.us for e in events])

    def single(self, n, mode=G["kGridFull"]):
        """n GRID commands one at a time (each waited for): microseconds each."""
        out = []
        for _ in range(n):
            handle = self.run.submit(
                self.handler, self.group, self.words(mode), exclusive=True
            )
            out.append(self.run.wait(handle, timeout=10.0).us)
        return np.array(out)

    def profile(self, session, prof):
        """Each core's profile page (summed over the commands so far): [cores,
        words]."""
        handle = self.run.submit("grid_prof", self.group, [prof.address])
        self.run.wait(handle, timeout=10.0)
        (pages,) = session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.group[1], : G["kGpWords"]]

    stop, failed, summary = Product.stop, Product.failed, Product.summary


def split_program(sp, tensors, fidelity="HiFi2"):
    """The K-split form's program for plan sp: every core gets the plan words."""
    words = sp.words()
    fp32 = "UnpackToDestFp32"
    return km.Program(
        "dense_split",
        kernels=SPLIT_KERNELS,
        buffers=[km.Buffer(*b) for b in sp.buffers()],
        args=lambda core: (words, words, words),
        compile_args={
            "reader": tensors["a"].accessor_args() + tensors["w"].accessor_args(),
            "writer": tensors["y"].accessor_args(),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {S["kCbArrive"]: fp32, S["kCbSum"]: fp32},
        },
        handlers=pl.SPLIT_H,
    )


class SplitProduct(GridProduct):
    """Plan sp (K split) loaded on a session with operand tensors a, w, y: one SPLIT
    command per call on cores 0 to sp.cores() - 1, exclusive (its ports), by `form`."""

    handler = "split"
    modes = {"full": S["kSplitFull"], "parts": S["kSplitParts"]}

    def __init__(
        self,
        session,
        sp,
        tensors,
        form="owner",
        fidelity="HiFi2",
        deadline=2.0,
        wait=1.0,
    ):
        self.g, self.t, self.form = sp, tensors, form
        self.group = (0, sp.cores())
        program = split_program(sp, tensors, fidelity)
        self.run = session.load(program, deadline=deadline, wait=wait)
        self.load_us = None

    def words(self, mode):
        addr = [self.t[k].address for k in ("a", "w", "y")]
        return pl.split_args(self.g, *addr, self.form, mode)
