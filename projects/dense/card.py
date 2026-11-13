"""Running the matrix product on a card through `komorebi.Session`: the program of a
plan, the operand tensors, and the records (LOAD, PRODUCT, PROF) issued through the
dispatch kernel on X280 cluster 0. Needs tt-metal's Python environment.
"""

import time

import numpy as np
import torch
import ttnn

import komorebi as km
from projects.dense import plan as pl

K = pl.K
KERNELS = pl.HERE / "kernels"


def bf16_tensor(device, x):
    """A float array (bf16 values) as a bf16 tile tensor in DRAM."""
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


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
        states = self.run.state()
        return {
            "kernel": self.run.result()["error_name"],
            "cores_ended": {
                k: sum(c[k] != 0 for c in states)
                for k in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {k: c[k] for k in ("core", "reader", "writer", "compute")}
                for c in states
                if c["reader"] or c["writer"] or c["compute"]
            ][:8],
        }
