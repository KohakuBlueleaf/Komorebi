"""Running the matrix product on a card through `komorebi.Session`: the program of a
plan, the operand tensors, and the records (LOAD, PRODUCT, PROF) issued through the
dispatch kernel on X280 cluster 0. Needs tt-metal's Python environment.
"""

import time

import numpy as np
import torch
import ttnn

import komorebi as km
from komorebi import _env, abi, layout
from projects.dense import plan as pl
from ref import dense as ref

K, G = pl.K, pl.G
KERNELS = pl.HERE / "kernels"
GRID_KERNELS = KERNELS / "grid"
LANE_FIELDS = abi.struct_fields(
    _env.include_dir() / "komorebi" / "device" / "km_lane.hpp", "LaneArgs"
)


def bf16_tensor(device, x):
    """A float array (bf16 values) as a bf16 tile tensor in DRAM."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def read_rows(session, y, rows, log=None):
    """Rows `rows` of the bf16 tile tensor Y as float32, read one row tile at a time
    (only the tiles holding them, at most 1 MB each), after the run's stop."""
    nc = tuple(y.padded_shape)[1] // ref.TILE
    tiles = np.unique(np.asarray(rows) // ref.TILE)
    got = {}
    for i, t in enumerate(tiles):
        part = session.read_tiles_bf16(y, int(t) * nc, nc)  # [nc, 32, 32]
        got[int(t)] = part.transpose(1, 0, 2).reshape(ref.TILE, nc * ref.TILE)
        if log:
            log(f"row tile {i + 1} of {len(tiles)} read")
    return np.stack([got[r // ref.TILE][r % ref.TILE] for r in rows])


def prof_tensor(device, cores):
    """Zeroed profile pages in DRAM, one page of kProfBytes per core (row-major)."""
    return ttnn.from_torch(
        torch.zeros((cores, K["kProfBytes"] // 4), dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def program(p, a_tensor, fidelity="HiFi2"):
    """The resident program of plan p: the three kernels and the plan's buffers."""
    ct = ttnn.TensorAccessorArgs(a_tensor).get_compile_time_args()
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
        segments=pl.DENSE_H,
    )


class Product:
    """Plan p loaded on a session's grid with operand tensors a, w, y: LOAD at the
    start (resident form), then PRODUCT calls and PROF on the team (0, count)."""

    def __init__(self, session, p, tensors, count, fidelity="HiFi2", deadline=2.0):
        self.p, self.t, self.team = p, tensors, (0, count)
        self.run = session.load(
            program(p, tensors["a"], fidelity), deadline=deadline, wait=1.0
        )
        self.load_us = None
        if p.resident:
            w = tensors["w"].buffer_address()
            ticket = self.run.issue("load", self.team, pl.load_args(p, w))
            self.load_us = self.run.wait(ticket, timeout=10.0).us

    def words(self, mode):
        t = self.t
        return pl.product_args(
            self.p,
            t["a"].buffer_address(),
            t["y"].buffer_address(),
            t["w"].buffer_address(),
            mode,
        )

    def calls(self, n, mode=K["kModeFull"], timeout=60.0):
        """n PRODUCT records back to back: (host seconds from the first issue to the
        last event, each record's issue-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (n, 1))
        t0 = time.perf_counter()
        tickets = self.run.issue_many("product", self.team[0], self.team[1], words)
        events = self.run.wait(list(tickets), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} records not done: {bad[0]}")
        return host_s, np.array([e.us for e in events])

    def single(self, n, mode=K["kModeFull"]):
        """n PRODUCT records one at a time (each waited for): microseconds each."""
        out = []
        for _ in range(n):
            ticket = self.run.issue("product", self.team, self.words(mode))
            out.append(self.run.wait(ticket, timeout=10.0).us)
        return np.array(out)

    def profile(self, session, prof):
        """Each core's profile page (summed over the records so far): [cores, words]."""
        ticket = self.run.issue("prof", self.team, [prof.buffer_address()])
        self.run.wait(ticket, timeout=10.0)
        (pages,) = session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.team[1], : K["kProfWords"]]

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
            out["lanes"] = lanes_at_stall(self.run.state())
            out |= self.summary()
        except km.KomorebiError as state_err:
            out["state"] = f"{type(state_err).__name__}: {state_err}"
        return out

    def summary(self):
        """The stopped run's kernel result, cores ended, give-ups and lane waits."""
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
            "lane_wait_cycles_max": [
                max(c["lanes"][lane]["wait_cycles"] for c in states) for lane in (0, 1)
            ],
            "lane_gave_up": [
                {"core": c["core"]} | area["gave_up"]
                for c in states
                for area in c["lanes"]
                if area["gave_up"]
            ][:8],
        }


def lanes_at_stall(states):
    """By lane: blocks handed over (min, max), the cores at the min (first 8), their
    VALID words, and the cores' records completed (min, max)."""
    out = {
        "completed": [
            min(c["completed"] for c in states),
            max(c["completed"] for c in states),
        ]
    }
    for lane in (0, 1):
        blocks = [c["lanes"][lane]["blocks"] for c in states]
        low = [c for c in states if c["lanes"][lane]["blocks"] == min(blocks)]
        out[f"lane{lane}"] = {
            "blocks": [min(blocks), max(blocks)],
            "at_min": [c["core"] for c in low][:8],
            "valid_at_min": [c["lanes"][lane]["valid"] for c in low][:8],
        }
    return out


def grid_lanes(g, nocs, team, along, noc, buffer):
    """Lane words by core for plan g: lanes along the rect's rows or columns, or a
    lane of one core each where the layout gives every core its own operand."""
    common = (nocs.grid, nocs.panels, team)
    at = (nocs.noc0, nocs.noc_xy)
    rect = g.rect(nocs.grid)
    out = layout.lane_args(*common, rect, along, noc, *at, buffer)
    if {"row": "rows", "column": "cols"}[along] != g.layout:
        return out
    gx = nocs.grid[0]
    for core in g.places(nocs.grid):
        y, x = divmod(core, gx)
        one = layout.lane_args(*common, (x, y, 1, 1), along, noc, *at, buffer)
        out[core] = one[core]
    return out


def grid_program(g, nocs, tensors, fidelity="HiFi2"):
    """The grid form's program for plan g on a grid whose NoCs are `nocs`: each core's
    lane words (row lane on NoC0, column lane on NoC1), place and plan words."""
    team = layout.Team(*g.team(nocs.grid))
    row = grid_lanes(
        g, nocs, team, "row", 0, (G["kLaneRow"], G["kCbRow"], g.wa, g.bm * g.kb)
    )
    col = grid_lanes(
        g, nocs, team, "column", 1, (G["kLaneCol"], G["kCbCol"], g.ww, g.kb * g.bn)
    )
    places = g.places(nocs.grid)
    idle = [0] * (len(LANE_FIELDS) + G["kCoreWords"])

    def args(core):
        if core not in team.members:
            return idle, idle, idle[: G["kCoreWords"]] + [g.wa, g.ww]
        m0, n0 = places.get(core, (0, 0))
        mine = [int(core in places), m0, n0, *g.words()]
        lanes = [[words[core][f] for f in LANE_FIELDS] for words in (row, col)]
        return lanes[0] + mine, lanes[1] + mine, mine + [g.wa, g.ww]

    accessor = ttnn.TensorAccessorArgs
    return km.Program(
        "dense_grid",
        kernels=GRID_KERNELS,
        buffers=[km.Buffer(*b) for b in g.buffers()],
        args=args,
        compile_args={
            "reader": accessor(tensors["a"]).get_compile_time_args(),
            "writer": accessor(tensors["w"]).get_compile_time_args()
            + accessor(tensors["y"]).get_compile_time_args(),
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "unpack_to_dest_mode": {G["kCbPart"]: "UnpackToDestFp32"},
        },
        segments=pl.GRID_H,
    )


class GridProduct:
    """Plan g (the grid form) loaded on a session with operand tensors a, w, y: one
    GRID record per call on the rectangle's team, exclusive (its lanes)."""

    seg = "grid"
    modes = {
        "full": G["kGridFull"],
        "lanes": G["kGridLanes"],
        "product": G["kGridProduct"],
    }

    def __init__(self, session, g, tensors, fidelity="HiFi2", deadline=2.0, wait=1.0):
        self.g, self.t = g, tensors
        nocs = session.nocs()
        self.team = g.team(nocs.grid)
        program = grid_program(g, nocs, tensors, fidelity)
        self.run = session.load(program, deadline=deadline, wait=wait)
        self.load_us = None

    def words(self, mode):
        t = self.t
        addr = [t[k].buffer_address() for k in ("a", "w", "y")]
        return pl.grid_args(self.g, *addr, mode)

    def calls(self, n, mode=G["kGridFull"], timeout=60.0):
        """n GRID records back to back: (host seconds from the first issue to the last
        event, each record's issue-to-completion microseconds)."""
        words = np.tile(np.array(self.words(mode), np.int64), (n, 1))
        t0 = time.perf_counter()
        first, count = self.team
        tickets = self.run.issue_many(self.seg, first, count, words, exclusive=True)
        events = self.run.wait(list(tickets), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} records not done: {bad[0]}")
        return host_s, np.array([e.us for e in events])

    def single(self, n, mode=G["kGridFull"]):
        """n GRID records one at a time (each waited for): microseconds each."""
        out = []
        for _ in range(n):
            ticket = self.run.issue(
                self.seg, self.team, self.words(mode), exclusive=True
            )
            out.append(self.run.wait(ticket, timeout=10.0).us)
        return np.array(out)

    def profile(self, session, prof):
        """Each core's profile page (summed over the records so far): [cores, words]."""
        ticket = self.run.issue("grid_prof", self.team, [prof.buffer_address()])
        self.run.wait(ticket, timeout=10.0)
        (pages,) = session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(-1, K["kProfBytes"] // 4)
        return words[: self.team[1], : G["kGpWords"]]

    stop, failed, summary = Product.stop, Product.failed, Product.summary
