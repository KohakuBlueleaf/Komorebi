"""Runs resident worker kernels on every core but one; the last core runs the feeder.

The host writes inputs and zeroed control blocks before the launch and reads results
only after every kernel returned."""

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import records as rec
import torch
import ttnn

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
    """A CoreRangeSet of single cores [(x, y), ...]."""
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores]
    )


def int32_tensor(device, rows, memory_config=None):
    """A row-major int32 tensor with the bits of the uint32 rows (one page a row)."""
    return ttnn.from_torch(
        torch.from_numpy(rec.as_int32(rows)),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def read_int32(tensor):
    """A tensor read back as a numpy int32 array."""
    return ttnn.to_torch(tensor).numpy().astype(np.int32)


@dataclass
class Kernels:
    """The worker's three kernels and what they take besides the start-up args."""

    reader: Path
    compute: Path
    writer: Path
    reader_ct: list = field(default_factory=list)
    writer_ct: list = field(default_factory=list)
    compute_config: object = None
    cbs: list = field(default_factory=list)  # (index, data format, page bytes, pages)
    defines: list = field(default_factory=list)


@dataclass
class Result:
    """What a run left: feeder summary and per-record rows, state words, trace."""

    summary: dict
    records: dict
    state: np.ndarray
    trace: np.ndarray | None
    mhz: int
    host_s: float


class Fed:
    """A device's grid split into workers (cores 0 to n - 2) and the feeder (last)."""

    def __init__(self, device):
        self.device = device
        g = device.compute_with_storage_grid_size()
        self.grid = (g.x, g.y)
        self.n_cores = g.x * g.y
        self.workers = self.n_cores - 1
        self.logical = [(i % g.x, i // g.x) for i in range(self.n_cores)]
        self.noc = []
        for x, y in self.logical:
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            self.noc.append(c.x | c.y << 16)
        self.worker_set = core_set(self.logical[: self.workers])
        self.feeder_set = core_set(self.logical[self.workers :])
        self.mhz = aiclk_mhz()

    def control_blocks(self):
        """Zeroed control blocks, one shard per core: the same L1 address everywhere."""
        gx, gy = self.grid
        words = K["KM_CTL_BYTES"] // 4
        config = ttnn.create_sharded_memory_config(
            (1, words),
            ttnn.CoreGrid(y=gy, x=gx),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return int32_tensor(self.device, np.zeros((self.n_cores, words)), config)

    def run(
        self,
        kernels,
        records,
        program_args=None,
        tensors=(),
        depth=4,
        deadline_s=0.05,
        wait_s=2.0,
        idle_s=10.0,
        trace_cap=0,
        send_exit=True,
        release=0,
    ):
        """Runs `records` (lists of 16 words) and returns a Result.

        program_args(core) gives the (reader, writer, compute) arguments that follow
        the start-up arguments."""
        dev, mhz = self.device, self.mhz
        n = len(records)
        tag = (random.getrandbits(16) | 1) & 0xFFFF
        ctl = self.control_blocks()
        ctl_addr = ctl.buffer_address()
        core_table = np.zeros((1, K["KM_CORES_MAX"]), dtype=np.uint64)
        core_table[0, : self.n_cores] = self.noc
        cores_t = int32_tensor(dev, core_table)
        table_t = int32_tensor(dev, np.asarray(records, dtype=np.uint64).reshape(n, 16))
        results_t = int32_tensor(dev, np.zeros((n + 1, 16)))
        trace_t = None
        if trace_cap:
            words = trace_cap * K["kTraceWords"]
            trace_t = int32_tensor(dev, np.zeros((self.workers, words)))
        log(f"tensors: ctl at {ctl_addr:#x}, {n} records, tag {tag:#06x}")

        grp, n_groups = rec.groups(self.grid, self.workers)
        feeder = self.workers
        base = {
            "ctl": ctl_addr,
            "cores": cores_t.buffer_address(),
            "disp_xy": self.noc[feeder],
            "ack_base": ctl_addr + K["KM_CTL_DONE"],
            "ready_base": ctl_addr + K["KM_CTL_BOOT"],
            "tag": tag,
            "depth": depth,
            "row": self.grid[0],
            "wait_limit": rec.limit_units(wait_s, mhz),
            "idle_limit": rec.limit_units(idle_s, mhz),
            "trace": trace_t.buffer_address() if trace_t is not None else 0,
            "trace_cap": trace_cap,
        }
        reader_args, writer_args, compute_args = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        for core in range(self.workers):
            x, y = self.logical[core]
            leader, rank, members, ready_slot = grp[core]
            start = rec.start_args(
                base
                | {
                    "me": core,
                    "mailbox": ctl_addr + K["KM_CTL_JOB"] if core == 0 else 0,
                    "leader": leader,
                    "rank": rank,
                    "members": members,
                    "ready_slot": ready_slot,
                }
            )
            extra = program_args(core) if program_args else ([], [], [])
            reader_args[x][y] = start + list(extra[0])
            writer_args[x][y] = start + list(extra[1])
            compute = [ctl_addr, core, base["wait_limit"], base["idle_limit"]]
            compute_args[x][y] = compute + list(extra[2])
        feeder_args = ttnn.RuntimeArgs()
        fx, fy = self.logical[feeder]
        fd = {
            "kFdCtl": ctl_addr,
            "kFdWorkers": self.workers,
            "kFdCores": cores_t.buffer_address(),
            "kFdTable": table_t.buffer_address(),
            "kFdRecords": n,
            "kFdDepth": depth,
            "kFdDeadline": rec.limit_units(deadline_s, mhz),
            "kFdResults": results_t.buffer_address(),
            "kFdGroups": n_groups,
            "kFdBootLimit": rec.limit_units(wait_s, mhz),
            "kFdExit": int(send_exit),
            "kFdRelease": release,
        }
        words = [0] * K["kFdArgs"]
        for name, value in fd.items():
            words[K[name]] = value
        feeder_args[fx][fy] = words

        include = [rec.INCLUDE]
        u32 = ttnn.uint32

        def cb(index, fmt, page, pages, cores):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index, data_format=fmt, page_size=page
                    )
                ],
            )

        tok = K["kTokenBytes"]
        cbs = [cb(i, f, p, c, self.worker_set) for i, f, p, c in kernels.cbs]
        cbs += [
            cb(K["kCbTokenCompute"], u32, tok, 1, self.worker_set),
            cb(K["kCbTokenWriter"], u32, tok, 1, self.worker_set),
            cb(K["kFdCbTable"], u32, 64 * (n + 1), 1, self.feeder_set),
            cb(K["kFdCbResults"], u32, 64 * (n + 1), 1, self.feeder_set),
        ]
        if trace_cap:
            page = trace_cap * K["kTraceWords"] * 4
            cbs.append(cb(K["kCbTrace"], u32, page, 1, self.worker_set))

        def dm(source, cores, args, riscv, ct=()):
            processor, noc = (
                (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
                if riscv == 0
                else (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
            )
            return ttnn.KernelDescriptor(
                kernel_source=str(source),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=list(ct),
                defines=list(kernels.defines),
                runtime_args=args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                compiler_include_paths=include,
            )

        descriptor = ttnn.ProgramDescriptor(
            kernels=[
                dm(kernels.reader, self.worker_set, reader_args, 0, kernels.reader_ct),
                dm(kernels.writer, self.worker_set, writer_args, 1, kernels.writer_ct),
                ttnn.KernelDescriptor(
                    kernel_source=str(kernels.compute),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=self.worker_set,
                    compile_time_args=[],
                    defines=list(kernels.defines),
                    runtime_args=compute_args,
                    config=kernels.compute_config or ttnn.ComputeConfigDescriptor(),
                    compiler_include_paths=include,
                ),
                dm(FEEDER_KERNEL, self.feeder_set, feeder_args, 0),
            ],
            semaphores=[],
            cbs=cbs,
        )
        io = [ctl, cores_t, table_t, *tensors, results_t]
        if trace_t is not None:
            io.insert(-1, trace_t)
        log("launch")
        t0 = time.perf_counter()
        ttnn.generic_op(io, descriptor)
        ttnn.synchronize_device(dev)
        host_s = time.perf_counter() - t0
        log(f"program done in {host_s:.2f}s")
        rows = read_int32(results_t)
        state = rec.state_words(read_int32(ctl))
        trace = None
        if trace_t is not None:
            trace = rec.as_uint32(read_int32(trace_t)).reshape(self.workers, -1, 4)
        log("results read back")
        return Result(
            summary=rec.feeder_summary(rows),
            records=rec.feeder_records(rows),
            state=state,
            trace=trace,
            mhz=mhz,
            host_s=host_s,
        )
