"""Running attention on a card through `komorebi.Session`: the program of a plan, its
broadcast words from `komorebi.layout`, the tensors, and the ATTEND commands submitted
through the dispatcher on X280 cluster 0. Needs tt-metal's Python environment."""

import time

import numpy as np

import komorebi as km
from komorebi import abi, layout
from projects.attention import plan as pl

K = pl.K
KERNELS = pl.HERE / "kernels"
BROADCAST_FIELDS = abi.struct_fields(abi.headers().device[-1], "BroadcastArgs")
BROADCAST_ID = 0


def bf16_tensor(session, values):
    """A float array of bf16 values [..., rows, cols] as a bf16 tile tensor in DRAM,
    flattened to [rows of all leading axes, cols] (the same page order)."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    return session.tensor(values, "bfloat16", "tile")


def table_tensor(session, words):
    """Job tables [cores, kTableBytes / 4] uint32 as a row-major DRAM tensor, one page
    per core."""
    return session.tensor(words, "uint32")


def prof_tensor(session, cores):
    """Profile pages [cores, kProfBytes / 4] uint32, zero, as a row-major DRAM tensor
    (one page of kProfBytes per core)."""
    return table_tensor(session, np.zeros((cores, K["kProfBytes"] // 4), np.uint32))


def broadcast_words(nocs, plan, group, noc=0):
    """Each core's km::BroadcastArgs words: broadcasts of plan.length cores down every
    grid column on NoC `noc` (the sender first in its order), one band of rows after
    the other."""
    grid_columns, grid_rows = nocs.grid
    buffer = (BROADCAST_ID, K["kCbKv"], plan.window, plan.block_pages())
    by_core = {}
    for top in range(0, grid_rows, plan.length):
        rect = (0, top, grid_columns, plan.length)
        broadcast_arguments = layout.broadcast_args(
            nocs.grid,
            nocs.panels,
            group,
            rect,
            "column",
            noc,
            nocs.noc0,
            nocs.noc_xy,
            buffer,
        )
        by_core.update(
            {
                core: arguments
                for core, arguments in broadcast_arguments.items()
                if arguments["rank"] != layout.BROADCAST_NONE
            }
        )
    return {
        core: [arguments[field] for field in BROADCAST_FIELDS]
        for core, arguments in by_core.items()
    }


def program(
    plan,
    q_tensor,
    o_tensor,
    scale,
    approx=False,
    fidelity="HiFi2",
    broadcasts=None,
    noc=0,
):
    """The persistent program of `plan`: the three kernels, the plan's buffers, each
    core's broadcast words (`broadcasts`, by core) for the kernel on NoC `noc` (0: the
    reader, 1: the writer) and the compute's window."""
    call = plan.dims
    scale_bits, threshold_bits = pl.exp_constants(scale, approx)
    tile_sizes = [call.dt, call.dvt, plan.r, plan.kb]
    defines = {"BROADCAST_ON_WRITER": "1"} if noc == 1 else {}
    if call.causal:
        defines["FLASH_CAUSAL"] = "1"
    if approx:
        defines["FLASH_APPROX"] = "1"

    def args(core):
        if noc == 1:
            return [], broadcasts[core], [plan.window]
        return broadcasts[core], [], [plan.window]

    return km.Program(
        "attention",
        kernels=KERNELS,
        buffers=[km.Buffer(*buffer) for buffer in plan.buffers()],
        args=args,
        compile_args={
            "reader": tile_sizes + [plan.window] + q_tensor.accessor_args(),
            "writer": tile_sizes + [plan.window] + o_tensor.accessor_args(),
            "compute": tile_sizes + [scale_bits, threshold_bits],
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
        handlers=pl.ATTENTION_H,
    )


class Attention:
    """A plan loaded on a session's grid with tensors q, k, v, o and the job tables:
    ATTEND commands on the group of every core."""

    def __init__(
        self, session, plan, tensors, scale, approx=False, deadline=2.0, wait=2.0, noc=0
    ):
        nocs = session.nocs()
        cores = len(nocs.words)
        if len(plan.jobs) != cores:
            raise ValueError(f"plan for {len(plan.jobs)} cores, grid has {cores}")
        self.plan, self.tensors, self.group = plan, tensors, (0, cores)
        broadcasts = broadcast_words(nocs, plan, layout.Group(0, cores), noc)
        persistent_program = program(
            plan,
            tensors["q"],
            tensors["o"],
            scale,
            approx,
            broadcasts=broadcasts,
            noc=noc,
        )
        self.run = session.load(persistent_program, deadline=deadline, wait=wait)

    def words(self, mode=K["kModeFull"], passes=1):
        tensors = self.tensors
        return pl.call_args(
            *(tensors[name].address for name in ("q", "k", "v", "o", "tables")),
            mode,
            passes,
        )

    def calls(self, count, mode=K["kModeFull"], passes=1, timeout=60.0):
        """`count` ATTEND commands back to back (each `passes` times over its jobs):
        (host seconds from the first submit to the last event, each command's
        microseconds)."""
        words = np.tile(np.array(self.words(mode, passes), np.int64), (count, 1))
        started = time.perf_counter()
        handles = self.run.submit_many(
            "attend", self.group[0], self.group[1], words, exclusive=True
        )
        events = self.run.wait(list(handles), timeout=timeout)
        host_s = time.perf_counter() - started
        unfinished = [event for event in events if event.kind != "done"]
        if unfinished:
            raise km.KomorebiError(
                f"{len(unfinished)} commands not done: {unfinished[0]}"
            )
        return host_s, np.array([event.microseconds for event in events])

    def profile(self, session, zero=True, timeout=10.0):
        """Every core's profile page since the last zeroing ([cores, kProfWords]
        uint32, attention.h kPr*), stored by a PROF command into tensors["prof"] (made
        before the load); the pages are zeroed after (`zero`)."""
        profile_tensor = self.tensors["prof"]
        words = [profile_tensor.address, 1 if zero else 0]
        handle = self.run.submit("prof", self.group, words)
        self.run.wait(handle, timeout=timeout)
        (pages,) = session.read(profile_tensor)
        words = pages.reshape(-1).view(np.uint32).reshape(self.group[1], -1)
        return words[:, : K["kProfWords"]]

    def stop(self):
        """Stops the run; the kernel's result, the cores that ended, the give-ups, and
        `ok`: no kernel error, no give-up, every core's three kernels ended."""
        self.run.stop()
        states = self.run.state()
        report = {
            "kernel": self.run.result()["error_name"],
            "cores": len(states),
            "cores_ended": {
                name: sum(state[name] != 0 for state in states)
                for name in ("reader_end", "writer_end", "compute_end")
            },
            "gave_up": [
                {name: state[name] for name in ("core", "reader", "writer", "compute")}
                for state in states
                if state["reader"] or state["writer"] or state["compute"]
            ][:8],
            "broadcast_blocks": sum(
                state["broadcasts"][BROADCAST_ID]["blocks"] for state in states
            ),
        }
        ended = all(count == len(states) for count in report["cores_ended"].values())
        report["ok"] = report["kernel"] == "none" and not report["gave_up"] and ended
        return report
