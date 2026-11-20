"""Runs persistent kernel kernels on every core but one; the last core runs the feeder.

The host writes inputs and zeroed control blocks before the launch and reads results
only after every kernel returned."""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from komorebi.device import chips, mock
from komorebi.program.several import compute_config
from komorebi.session import kernels as kernel_module
from komorebi.session import process
from komorebi.tools.feeder import commands as feeder_commands

CONSTANTS = feeder_commands.CONSTANTS
FEEDER_KERNEL = feeder_commands.HERE / "feeder.cpp"


STARTED = time.monotonic()


def log(message):
    """One progress line with seconds since start, flushed (the watcher reads it)."""
    print(f"[{time.monotonic() - STARTED:7.2f}] {message}", flush=True)


def aiclk_mhz(device=None):
    """AICLK of the open `device` (its driver's number: Device.driver_device), else of
    the card TT_VISIBLE_DEVICES names, from the driver's sysfs (MHz)."""
    number = device.driver_device()[0] if device is not None else -1
    return process.aiclk_mhz(driver=number if number >= 0 else None)


def core_set(cores):
    """Single cores [(x, y), ...] as (x0, y0, x1, y1) rectangles."""
    return [(x, y, x, y) for x, y in cores]


def int32_tensor(session, rows, shard_grid=None):
    """A row-major int32 tensor with the bits of the uint32 rows (one page a row)."""
    return session.tensor(
        feeder_commands.as_int32(rows), "int32", shard_grid=shard_grid
    )


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
    opt: dict = field(default_factory=dict)  # a kernel's level by name, as Program.opt


@dataclass
class Result:
    """What a run left: feeder summary and per-command rows, state words, trace."""

    summary: dict
    commands: dict
    state: np.ndarray
    ports: np.ndarray  # [cores, KM_PORTS, 2]: count, wanted
    trace: np.ndarray | None
    mhz: float  # the wall clock's ticks a microsecond (chips.tick_mhz)
    host_s: float
    control: np.ndarray | None = None  # every core's control block, uint32 words
    lines: np.ndarray | None = None  # dynamic run: [workers, 4] status lines
    rows: np.ndarray | None = None  # the feeder's result rows, uint32
    aiclk: int | None = 0  # the clock the chip reported (tt_aiclk), MHz; None: not read


class Fed:
    """An open session's grid split into workers (cores 0 to n - 2) and the feeder
    (last); the session runs no program of its own."""

    def __init__(self, session):
        self.session = session
        grid_x, grid_y = session.grid
        self.grid = (grid_x, grid_y)
        self.core_count = grid_x * grid_y
        self.workers = self.core_count - 1
        self.logical = [(i % grid_x, i // grid_x) for i in range(self.core_count)]
        self.noc = []
        for x, y in self.logical:
            noc_x, noc_y = session.device.worker_core(x, y)
            self.noc.append(noc_x | noc_y << 16)
        self.worker_set = core_set(self.logical[: self.workers])
        self.feeder_set = core_set(self.logical[self.workers :])
        if mock.active():  # a prebuild's stand-in card: no card's clock to read
            log("aiclk not read: the prebuild's stand-in card")
            self.aiclk, self.mhz = None, mock.mhz(session.device.arch())
        else:
            self.aiclk = aiclk_mhz(session.device)
            # the ticks a microsecond: what cycles become time by, and limits cycles by
            self.mhz = chips.tick_mhz(session.device.arch(), self.aiclk)

    def control_blocks(self):
        """Zeroed control blocks, one shard per core: the same L1 address everywhere."""
        return self.l1_blocks(CONSTANTS["KM_CONTROL_BYTES"] // 4)

    def l1_blocks(self, words):
        """A zeroed L1 area of `words` words on every core, at one address."""
        return int32_tensor(self.session, np.zeros((self.core_count, words)), self.grid)

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
        command_count = len(commands)
        tag = (random.getrandbits(16) | 1) & 0xFFFF
        control = self.control_blocks()
        control_addr = control.address
        core_table = np.zeros((1, CONSTANTS["KM_CORES_MAX"]), dtype=np.uint64)
        core_table[0, : self.core_count] = self.noc
        cores_t = int32_tensor(session, core_table)
        table_t = int32_tensor(
            session, np.asarray(commands, dtype=np.uint64).reshape(command_count, 16)
        )
        results_t = int32_tensor(
            session, np.zeros((command_count + 2, 16))
        )  # + EXIT, end command
        if trace_cap % 4:  # the kernel writes cap * 16 bytes as one page
            raise ValueError(f"trace_cap {trace_cap} is not a multiple of 4")
        trace_t = None
        if trace_cap:
            words = trace_cap * CONSTANTS["kTraceWords"]
            trace_t = int32_tensor(session, np.zeros((self.workers, words)))
        messages_rw, messages_c, queue_t, lines_t = [], [], None, None
        if dynamic is not None:
            dynamic.check(self.workers, queue_bytes, depth)
            page = 32 * dynamic.message_bytes  # kMessagesPageMessages messages a page
            shape = (self.workers * dynamic.queue_messages // 32, page // 4)
            queue_t = session.empty(shape, "int32")
            lines_t = self.l1_blocks(
                2 * CONSTANTS["KM_CORES_MAX"] * 4
            )  # lines, first pass
            messages_rw = dynamic.args(queue_t.address, lines_t.address)
            messages_c = dynamic.compute_args(self.workers)
            log(f"dynamic: {dynamic.queue_bytes(self.workers) >> 20} MiB of queues")
        log(
            f"tensors: control at {control_addr:#x}, {command_count} commands, "
            f"tag {tag:#06x}"
        )

        grp, n_groups = feeder_commands.groups(self.grid, self.workers)
        feeder = self.workers
        base = {
            "control": control_addr,
            "cores": cores_t.address,
            "dispatcher_coordinates": self.noc[feeder],
            "ack_base": control_addr + CONSTANTS["KM_CONTROL_DONE"],
            "ready_base": control_addr + CONSTANTS["KM_CONTROL_BOOT"],
            "tag": tag,
            "depth": depth,
            "workers": self.workers,
            "row": self.grid[0] if row is None else row,
            "wait_limit": feeder_commands.limit_units(wait_s, mhz),
            "idle_limit": (
                feeder_commands.limit_units(idle_s, mhz) if idle_s else 0
            ),  # 0: none
            "trace": trace_t.address if trace_t is not None else 0,
            "trace_cap": trace_cap,
        }
        mailbox = control_addr + CONSTANTS["KM_CONTROL_JOB"]
        aligned = [control_addr, base["ack_base"], base["ready_base"], mailbox]
        if any(address % 16 for address in aligned):  # small writes go slot to slot
            raise ValueError(
                f"bases not 16-byte aligned: {[hex(address) for address in aligned]}"
            )
        reader_args, writer_args, compute_args = [], [], []
        for core in range(self.workers):
            x, y = self.logical[core]
            leader, rank, peers, ready_slot = grp[core]
            start = feeder_commands.start_args(
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
            reader_args.append((x, y, start + messages_rw + list(extra[0])))
            writer_args.append((x, y, start + messages_rw + list(extra[1])))
            compute = [control_addr, core, base["wait_limit"], base["idle_limit"]]
            compute_args.append((x, y, compute + messages_c + list(extra[2])))
        feeder_x, feeder_y = self.logical[feeder]
        feeder_words = {
            "kFdControl": control_addr,
            "kFdWorkers": self.workers,
            "kFdCores": cores_t.address,
            "kFdTable": table_t.address,
            "kFdCommands": command_count,
            "kFdDepth": depth,
            "kFdDeadline": feeder_commands.limit_units(deadline_s, mhz),
            "kFdResults": results_t.address,
            "kFdGroups": n_groups,
            "kFdBootLimit": feeder_commands.limit_units(wait_s, mhz),
            "kFdExit": int(send_exit),
            "kFdRelease": release,
            "kFdRaw": raw,
            "kFdMessagesHandler": dynamic.word() if dynamic is not None else 0,
            "kFdLines": lines_t.address if lines_t is not None else 0,
        }
        words = [0] * CONSTANTS["kFdArgs"]
        for name, value in feeder_words.items():
            words[CONSTANTS[name]] = value
        feeder_args = [(feeder_x, feeder_y, words)]

        include = [str(feeder_commands.INCLUDE)]
        launch = session.launch()
        tok = CONSTANTS["kTokenBytes"]
        cbs = [
            (i, data_format, page_bytes, page_count, self.worker_set)
            for i, data_format, page_bytes, page_count in kernels.cbs
        ]
        cbs += [
            (CONSTANTS["kCbTokenCompute"], "uint32", tok, 1, self.worker_set),
            (CONSTANTS["kCbTokenWriter"], "uint32", tok, 1, self.worker_set),
            (
                CONSTANTS["kFdCbTable"],
                "uint32",
                64 * (command_count + 2),
                1,
                self.feeder_set,
            ),
            (
                CONSTANTS["kFdCbResults"],
                "uint32",
                64 * (command_count + 2),
                1,
                self.feeder_set,
            ),
        ]
        if trace_cap:
            page = trace_cap * CONSTANTS["kTraceWords"] * 4
            cbs.append((CONSTANTS["kCbTrace"], "uint32", page, 1, self.worker_set))
        # a pad first where the buffers' base is not on a line (Wormhole's): the
        # buffers after it start 64-byte aligned, as the messages' reads need
        pad = -(self.session.l1_bytes - self.session.l1()[1]) % 64
        if pad:
            index = min(set(range(32)) - {cb[0] for cb in cbs})
            launch.cb(self.worker_set, index, "uint32", pad, pad, first=True)
        for index, fmt, page, pages, cores in cbs:
            launch.cb(cores, index, fmt, page, page * pages)
        # the session's own defines for a load of one card (its island, never split)
        defines = list(kernels.defines)
        defines += kernel_module.one_island(self.session.island, None)
        worker = [
            (kernels.reader, reader_args, 0, kernels.reader_ct, {}),
            (kernels.writer, writer_args, 1, kernels.writer_ct, {}),
            (
                kernels.compute,
                compute_args,
                2,
                [],
                compute_config(
                    kernels.compute_options,
                    self.session.device.buffer_indices(),
                    self.session.device.arch(),
                ),
            ),
        ]
        names = ("reader", "writer", "compute")
        for source, args, processor, compile_arguments, config in worker:
            build = kernel_module.chip_build(
                defines,
                kernels.opt.get(names[processor]),
                names[processor],
                self.session.device.arch(),
            )
            launch.kernel(
                str(source), self.worker_set, processor, list(compile_arguments),
                includes=include, args=args, **build, **config,
            )  # fmt: skip
        launch.kernel(
            str(FEEDER_KERNEL), self.feeder_set, 0, [], defines, include, feeder_args
        )
        log("launch")
        started = time.perf_counter()
        launch.run()
        session.device.finish()
        host_s = time.perf_counter() - started
        log(f"program done in {host_s:.2f}s")
        rows = read_int32(results_t)
        control_words = read_int32(control)
        state, ports = feeder_commands.state_words(
            control_words
        ), feeder_commands.port_words(control_words)
        lines = None
        if lines_t is not None:  # the feeder's shard holds the lines
            shard = feeder_commands.as_uint32(read_int32(lines_t))[feeder]
            lines = shard[: self.workers * 4].reshape(self.workers, 4)
        trace = None
        if trace_t is not None:
            trace = feeder_commands.as_uint32(read_int32(trace_t)).reshape(
                self.workers, -1, 4
            )
        log("results read back")
        return Result(
            summary=feeder_commands.feeder_summary(rows),
            commands=feeder_commands.feeder_commands(rows[: command_count + 1]),
            state=state,
            ports=ports,
            trace=trace,
            mhz=mhz,
            aiclk=self.aiclk,
            host_s=host_s,
            control=feeder_commands.as_uint32(control_words),
            lines=lines,
            rows=feeder_commands.as_uint32(rows),
        )
