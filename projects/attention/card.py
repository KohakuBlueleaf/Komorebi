"""Running attention on a card through `komorebi.Session`: the program of a plan, its
lane words from `komorebi.layout`, the tensors, and the ATTEND records issued through
the dispatch kernel on X280 cluster 0. Needs tt-metal's Python environment.
"""

import time

import numpy as np

import komorebi as km
from komorebi import abi, layout
from projects.attention import plan as pl

K = pl.K
KERNELS = pl.HERE / "kernels"
LANE_FIELDS = abi.struct_fields(abi.headers().device[-1], "LaneArgs")
LANE_ID = 0


def bf16_tensor(session, x):
    """A float array of bf16 values [..., rows, cols] as a bf16 tile tensor in DRAM,
    flattened to [rows of all leading axes, cols] (the same page order)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
    return session.tensor(x, "bfloat16", "tile")


def table_tensor(session, words):
    """Job tables [cores, kTableBytes / 4] uint32 as a row-major DRAM tensor, one page
    per core."""
    return session.tensor(words, "uint32")


def prof_tensor(session, cores):
    """Profile pages [cores, kProfBytes / 4] uint32, zero, as a row-major DRAM tensor
    (one page of kProfBytes per core)."""
    return table_tensor(session, np.zeros((cores, K["kProfBytes"] // 4), np.uint32))


def lane_words(nocs, p, team, noc=0):
    """Each core's km::LaneArgs words: lanes of p.length cores down every grid column
    on NoC `noc` (the sender first in its order), one band of rows after the other."""
    gx, gy = nocs.grid
    buffer = (LANE_ID, K["kCbKv"], p.window, p.block_pages())
    out = {}
    for y0 in range(0, gy, p.length):
        rect = (0, y0, gx, p.length)
        args = layout.lane_args(
            nocs.grid,
            nocs.panels,
            team,
            rect,
            "column",
            noc,
            nocs.noc0,
            nocs.noc_xy,
            buffer,
        )
        out.update({c: a for c, a in args.items() if a["rank"] != layout.LANE_NONE})
    return {c: [a[f] for f in LANE_FIELDS] for c, a in out.items()}


def program(
    p, q_tensor, o_tensor, scale, approx=False, fidelity="HiFi2", lanes=None, noc=0
):
    """The resident program of plan p: the three kernels, the plan's buffers, each
    core's lane words (`lanes`, by core) for the kernel on NoC `noc` (0: the reader, 1:
    the writer) and the compute's window."""
    s = p.dims
    a_bits, thr_bits = pl.exp_constants(scale, approx)
    shape = [s.dt, s.dvt, p.r, p.kb]
    defines = {"LANE_ON_WRITER": "1"} if noc == 1 else {}
    if s.causal:
        defines["FLASH_CAUSAL"] = "1"
    if approx:
        defines["FLASH_APPROX"] = "1"

    def args(core):
        if noc == 1:
            return [], lanes[core], [p.window]
        return lanes[core], [], [p.window]

    return km.Program(
        "attention",
        kernels=KERNELS,
        buffers=[km.Buffer(*b) for b in p.buffers()],
        args=args,
        compile_args={
            "reader": shape + [p.window] + q_tensor.accessor_args(),
            "writer": shape + [p.window] + o_tensor.accessor_args(),
            "compute": shape + [a_bits, thr_bits],
        },
        compute={
            "math_fidelity": fidelity,
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": False,
            "math_approx_mode": approx,
            "unpack_to_dest_mode": {
                K["kCbO"]: "UnpackToDestFp32",
                K["kCbSt"]: "UnpackToDestFp32",
            },
        },
        defines=defines,
        segments=pl.ATTENTION_H,
    )


class Attention:
    """Plan p loaded on a session's grid with tensors q, k, v, o and the job tables:
    ATTEND records on the team of every core."""

    def __init__(
        self, session, p, tensors, scale, approx=False, deadline=2.0, wait=2.0, noc=0
    ):
        nocs = session.nocs()
        cores = len(nocs.words)
        if len(p.jobs) != cores:
            raise ValueError(f"plan for {len(p.jobs)} cores, grid has {cores}")
        self.p, self.t, self.team = p, tensors, (0, cores)
        lanes = lane_words(nocs, p, layout.Team(0, cores), noc)
        prog = program(
            p, tensors["q"], tensors["o"], scale, approx, lanes=lanes, noc=noc
        )
        self.run = session.load(prog, deadline=deadline, wait=wait)

    def words(self, mode=K["kModeFull"], passes=1):
        t = self.t
        return pl.call_args(
            *(t[n].address for n in ("q", "k", "v", "o", "tables")),
            mode,
            passes,
        )

    def calls(self, n, mode=K["kModeFull"], passes=1, timeout=60.0):
        """n ATTEND records back to back (each `passes` times over its jobs): (host
        seconds from the first issue to the last event, each record's microseconds)."""
        words = np.tile(np.array(self.words(mode, passes), np.int64), (n, 1))
        t0 = time.perf_counter()
        tickets = self.run.issue_many(
            "attend", self.team[0], self.team[1], words, exclusive=True
        )
        events = self.run.wait(list(tickets), timeout=timeout)
        host_s = time.perf_counter() - t0
        bad = [e for e in events if e.kind != "done"]
        if bad:
            raise km.KomorebiError(f"{len(bad)} records not done: {bad[0]}")
        return host_s, np.array([e.us for e in events])

    def profile(self, session, zero=True, timeout=10.0):
        """Every core's profile page since the last zeroing ([cores, kProfWords]
        uint32, attention.h kPr*), stored by a PROF record into tensors["prof"] (made
        before the load); the pages are zeroed after (`zero`)."""
        prof = self.t["prof"]
        words = [prof.address, 1 if zero else 0]
        ticket = self.run.issue("prof", self.team, words)
        self.run.wait(ticket, timeout=timeout)
        (pages,) = session.read(prof)
        words = pages.reshape(-1).view(np.uint32).reshape(self.team[1], -1)
        return words[:, : K["kProfWords"]]

    def stop(self):
        """Stops the run; the kernel's result, the cores that ended, the give-ups, and
        `ok`: no kernel error, no give-up, every core's three kernels ended."""
        self.run.stop()
        states = self.run.state()
        out = {
            "kernel": self.run.result()["error_name"],
            "cores": len(states),
            "cores_ended": {
                k: sum(c[k] != 0 for c in states)
                for k in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {k: c[k] for k in ("core", "reader", "writer", "compute")}
                for c in states
                if c["reader"] or c["writer"] or c["compute"]
            ][:8],
            "lane_blocks": sum(c["lanes"][LANE_ID]["blocks"] for c in states),
        }
        ended = all(n == len(states) for n in out["cores_ended"].values())
        out["ok"] = out["kernel"] == "none" and not out["gave_up"] and ended
        return out
