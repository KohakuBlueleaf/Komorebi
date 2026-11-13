"""Runs persistent kernel kernels on every core but one; the last core runs the feeder.

The host writes inputs and zeroed control blocks before the launch and reads results
only after every kernel returned."""

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import commands as rec
import numpy as np

from komorebi.session import compute_config

K = rec.K
FEEDER_KERNEL = rec.HERE / "feeder.cpp"


T0 = time.monotonic()


def log(msg):
    """One progress line with seconds since start, flushed (the watcher reads it)."""
    print(f"[{time.monotonic() - T0:7.2f}] {msg}", flush=True)


def aiclk_mhz():
    """AICLK of the card this process uses, from the driver's sysfs (MHz)."""
    card = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def core_set(cores):
    """Single cores [(x, y), ...] as (x0, y0, x1, y1) rectangles."""
    return [(x, y, x, y) for x, y in cores]


def int32_tensor(session, rows, shard_grid=None):
    """A row-major int32 tensor with the bits of the uint32 rows (one page a row)."""
    return session.tensor(rec.as_int32(rows), "int32", shard_grid=shard_grid)


def read_int32(tensor):
    """A tensor read back as a numpy int32 array."""
    return tensor.read()


@dataclass
class Kernels:
    """The worker's three kernels and what they take besides the start-up args."""

    reader: Path
    compute: Path
    writer: Path
    reader_ct: list = field(default_factory=list)
    writer_ct: list = field(default_factory=list)
    compute_options: dict = field(default_factory=dict)  # as Program.compute
    cbs: list = field(default_factory=list)  # (index, format name, page bytes, pages)
    defines: list = field(default_factory=list)


@dataclass
class Result:
    """What a run left: feeder summary and per-command rows, state words, trace."""

    summary: dict
    commands: dict
    state: np.ndarray
    ports: np.ndarray  # [cores, KM_PORTS, 2]: count, wanted
    trace: np.ndarray | None
    mhz: int
    host_s: float
    ctl: np.ndarray | None = None  # every core's control block, uint32 words
    lines: np.ndarray | None = None  # dynamic run: [workers, 4] status lines
    rows: np.ndarray | None = None  # the feeder's result rows, uint32


class Fed:
    """An open session's grid split into workers (cores 0 to n - 2) and the feeder
    (last); the session runs no program of its own."""

    def __init__(self, session):
        self.session = session
        gx, gy = session.grid
        self.grid = (gx, gy)
        self.n_cores = gx * gy
        self.workers = self.n_cores - 1
        self.logical = [(i % gx, i // gx) for i in range(self.n_cores)]
        self.noc = []
        for x, y in self.logical:
            cx, cy = session.device.worker_core(x, y)
            self.noc.append(cx | cy << 16)
        self.worker_set = core_set(self.logical[: self.workers])
        self.feeder_set = core_set(self.logical[self.workers :])
        self.mhz = aiclk_mhz()

    def control_blocks(self):
        """Zeroed control blocks, one shard per core: the same L1 address everywhere."""
        return self.l1_blocks(K["KM_CTL_BYTES"] // 4)

    def l1_blocks(self, words):
        """A zeroed L1 area of `words` words on every core, at one address."""
        return int32_tensor(self.session, np.zeros((self.n_cores, words)), self.grid)

    def run(
        self,
        kernels,
        commands,
        program_args=None,
        tensors=(),
        depth=4,
        deadline_s=0.05,
        wait_s=2.0,
        idle_s=0.0,
        trace_cap=0,
        send_exit=True,
        release=0,
        row=None,
        raw=0,
        dynamic=None,
        queue_bytes=4 << 30,
    ):
        """Runs `commands` (lists of 16 words) and returns a Result. program_args(core):
        (reader, writer, compute) arguments after the start-up ones. `dynamic`
        (komorebi.program.Dynamic): its handler runs dynamic, queues in DRAM (at most
        queue_bytes), status lines on the feeder, its arguments before the program's."""
        session, mhz = self.session, self.mhz
        n = len(commands)
        tag = (random.getrandbits(16) | 1) & 0xFFFF
        ctl = self.control_blocks()
        ctl_addr = ctl.address
        core_table = np.zeros((1, K["KM_CORES_MAX"]), dtype=np.uint64)
        core_table[0, : self.n_cores] = self.noc
        cores_t = int32_tensor(session, core_table)
        table_t = int32_tensor(
            session, np.asarray(commands, dtype=np.uint64).reshape(n, 16)
        )
        results_t = int32_tensor(session, np.zeros((n + 2, 16)))  # + EXIT, end command
        if trace_cap % 4:  # the kernel writes cap * 16 bytes as one page
            raise ValueError(f"trace_cap {trace_cap} is not a multiple of 4")
        trace_t = None
        if trace_cap:
            words = trace_cap * K["kTraceWords"]
            trace_t = int32_tensor(session, np.zeros((self.workers, words)))
        dyn_rw, dyn_c, queue_t, lines_t = [], [], None, None
        if dynamic is not None:
            dynamic.check(self.workers, queue_bytes, depth)
            page = 32 * dynamic.message_bytes  # kDynPageMessages items a page
            shape = (self.workers * dynamic.queue_messages // 32, page // 4)
            queue_t = session.empty(shape, "int32")
            lines_t = self.l1_blocks(2 * K["KM_CORES_MAX"] * 4)  # lines, first pass
            dyn_rw = dynamic.args(queue_t.address, lines_t.address)
            dyn_c = dynamic.compute_args(self.workers)
            log(f"dynamic: {dynamic.queue_bytes(self.workers) >> 20} MiB of queues")
        log(f"tensors: ctl at {ctl_addr:#x}, {n} commands, tag {tag:#06x}")

        grp, n_groups = rec.groups(self.grid, self.workers)
        feeder = self.workers
        base = {
            "ctl": ctl_addr,
            "cores": cores_t.address,
            "disp_xy": self.noc[feeder],
            "ack_base": ctl_addr + K["KM_CTL_DONE"],
            "ready_base": ctl_addr + K["KM_CTL_BOOT"],
            "tag": tag,
            "depth": depth,
            "workers": self.workers,
            "row": self.grid[0] if row is None else row,
            "wait_limit": rec.limit_units(wait_s, mhz),
            "idle_limit": rec.limit_units(idle_s, mhz) if idle_s else 0,  # 0: none
            "trace": trace_t.address if trace_t is not None else 0,
            "trace_cap": trace_cap,
        }
        mailbox = ctl_addr + K["KM_CTL_JOB"]
        aligned = [ctl_addr, base["ack_base"], base["ready_base"], mailbox]
        if any(addr % 16 for addr in aligned):  # small writes go slot to slot
            raise ValueError(f"bases not 16-byte aligned: {[hex(a) for a in aligned]}")
        reader_args, writer_args, compute_args = [], [], []
        for core in range(self.workers):
            x, y = self.logical[core]
            leader, rank, peers, ready_slot = grp[core]
            start = rec.start_args(
                base
                | {
                    "me": core,
                    "mailbox": mailbox if core == 0 else 0,
                    "leader": leader,
                    "rank": rank,
                    "peers": peers,
                    "ready_slot": ready_slot,
                }
            )
            extra = program_args(core) if program_args else ([], [], [])
            reader_args.append((x, y, start + dyn_rw + list(extra[0])))
            writer_args.append((x, y, start + dyn_rw + list(extra[1])))
            compute = [ctl_addr, core, base["wait_limit"], base["idle_limit"]]
            compute_args.append((x, y, compute + dyn_c + list(extra[2])))
        fx, fy = self.logical[feeder]
        fd = {
            "kFdCtl": ctl_addr,
            "kFdWorkers": self.workers,
            "kFdCores": cores_t.address,
            "kFdTable": table_t.address,
            "kFdCommands": n,
            "kFdDepth": depth,
            "kFdDeadline": rec.limit_units(deadline_s, mhz),
            "kFdResults": results_t.address,
            "kFdGroups": n_groups,
            "kFdBootLimit": rec.limit_units(wait_s, mhz),
            "kFdExit": int(send_exit),
            "kFdRelease": release,
            "kFdRaw": raw,
            "kFdDynHandler": dynamic.handler if dynamic is not None else 0,
            "kFdLines": lines_t.address if lines_t is not None else 0,
        }
        words = [0] * K["kFdArgs"]
        for name, value in fd.items():
            words[K[name]] = value
        feeder_args = [(fx, fy, words)]

        include = [str(rec.INCLUDE)]
        launch = session.launch()
        tok = K["kTokenBytes"]
        cbs = [(i, f, p, c, self.worker_set) for i, f, p, c in kernels.cbs]
        cbs += [
            (K["kCbTokenCompute"], "uint32", tok, 1, self.worker_set),
            (K["kCbTokenWriter"], "uint32", tok, 1, self.worker_set),
            (K["kFdCbTable"], "uint32", 64 * (n + 2), 1, self.feeder_set),
            (K["kFdCbResults"], "uint32", 64 * (n + 2), 1, self.feeder_set),
        ]
        if trace_cap:
            page = trace_cap * K["kTraceWords"] * 4
            cbs.append((K["kCbTrace"], "uint32", page, 1, self.worker_set))
        for index, fmt, page, pages, cores in cbs:
            launch.cb(cores, index, fmt, page, page * pages)
        defines = list(kernels.defines)
        worker = [
            (kernels.reader, reader_args, 0, kernels.reader_ct, {}),
            (kernels.writer, writer_args, 1, kernels.writer_ct, {}),
            (
                kernels.compute,
                compute_args,
                2,
                [],
                compute_config(kernels.compute_options),
            ),
        ]
        for source, args, processor, ct, config in worker:
            launch.kernel(
                str(source), self.worker_set, processor, list(ct), defines, include,
                args, **config,
            )  # fmt: skip
        launch.kernel(
            str(FEEDER_KERNEL), self.feeder_set, 0, [], defines, include, feeder_args
        )
        log("launch")
        t0 = time.perf_counter()
        launch.run()
        session.device.finish()
        host_s = time.perf_counter() - t0
        log(f"program done in {host_s:.2f}s")
        rows = read_int32(results_t)
        ctl_words = read_int32(ctl)
        state, ports = rec.state_words(ctl_words), rec.port_words(ctl_words)
        lines = None
        if lines_t is not None:  # the feeder's shard holds the lines
            shard = rec.as_uint32(read_int32(lines_t))[feeder]
            lines = shard[: self.workers * 4].reshape(self.workers, 4)
        trace = None
        if trace_t is not None:
            trace = rec.as_uint32(read_int32(trace_t)).reshape(self.workers, -1, 4)
        log("results read back")
        return Result(
            summary=rec.feeder_summary(rows),
            commands=rec.feeder_commands(rows[: n + 1]),
            state=state,
            ports=ports,
            trace=trace,
            mhz=mhz,
            host_s=host_s,
            ctl=rec.as_uint32(ctl_words),
            lines=lines,
            rows=rec.as_uint32(rows),
        )
