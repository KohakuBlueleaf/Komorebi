"""A card session: opens the device, loads a program as resident workers on every
Tensix core and the dispatch kernel on X280 cluster 0, and hands out a `Run`.

Needs ttnn and a tt-metal tree; `import komorebi` alone does not import this module.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

from komorebi import abi, native
from komorebi.errors import ProgramError, StartError
from komorebi.run import Run

CTL_ALIGN = 64  # the dispatch kernel stores whole 64-byte lines into the control block
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI ack word in the cluster's DRAM bank
STATE_PAGES = 32  # pages of KMD_STATE_BYTES, interleaved: several in every DRAM bank


def aiclk_mhz(card=None):
    """The Tensix clock of the card this process uses, from the driver (MHz)."""
    card = card if card is not None else visible_card()
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    return int(path.read_text().split()[0])


def visible_card():
    return int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])


def _state_area(device):
    """A DRAM buffer whose bank address holds the dispatch kernel's state in every
    bank (KMD_STATE_BYTES at least); refused if it covers the runtime's IPI word."""
    h = abi.headers()
    rows = _int32_tensor(device, np.zeros((STATE_PAGES, h["KMD_STATE_BYTES"] // 4)))
    at = rows.buffer_address()
    if at <= IPI_ACK_DDR < at + STATE_PAGES * h["KMD_STATE_BYTES"]:
        raise StartError(f"the kernel state at {at:#x} covers the X280 IPI word")
    return rows


def _int32_tensor(device, rows, memory_config=None):
    """A row-major int32 tensor holding the bits of the uint32 rows."""
    data = np.asarray(rows, dtype=np.uint64).astype(np.uint32).view(np.int32)
    return ttnn.from_torch(
        torch.from_numpy(np.ascontiguousarray(data)),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_uint32(tensor):
    return ttnn.to_torch(tensor).numpy().astype(np.int64) & abi.M32


class Session:
    """One card. `card` selects it (TT_VISIBLE_DEVICES); `device` uses an open one.

    As a context manager it stops a run still loaded at the exit (on an exception
    through the runtime, without EXIT) and closes the device it opened.
    """

    def __init__(self, card=None, device=None, pll_mhz=1750, tree=None, log=None):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = pll_mhz
        self.tree = tree
        self.log = log or (lambda msg: None)
        self.run = None

    def __enter__(self):
        if self.device is None:
            self.open()
        return self

    def __exit__(self, kind, error, tb):
        try:
            if self.run is not None and not self.run.stopped:
                if kind is None:
                    self.run.stop()
                else:
                    self.run.abort()
        finally:
            self.close()

    def open(self):
        """Opens the card (slow dispatch) if the session was not given a device."""
        if self.card is not None:
            visible = os.environ.get("TT_VISIBLE_DEVICES")
            if visible is not None and visible.split(",")[0] != str(self.card):
                raise ValueError(f"card {self.card} is not visible ({visible})")
            os.environ["TT_VISIBLE_DEVICES"] = str(self.card)
        os.environ.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
        self.device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        self.device.disable_and_clear_program_cache()
        self._own_device = True

    def close(self):
        if self.device is not None and self._own_device:
            ttnn.close_mesh_device(self.device)
            self.device = None

    def load(
        self,
        program,
        depth=None,
        trace=0,
        deadline=1.0,
        wait=2.0,
        idle=0.0,
        kernel_idle=0.0,
        start=10.0,
        host_depth=64,
        event_depth=128,
        batch=64,
        settle_us=1.0,
        image=None,
    ):
        """Launches `program` on every core, then the dispatch kernel; returns a Run.
        The arguments are described in docs/reference/host.md (trace capacity, record
        deadline, the cores' wait bounds, ring depths, batch, settle pause).
        """
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        image = Path(image) if image else native.image_path(self.tree)
        module = native.load_module(self.tree)
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        mhz = aiclk_mhz(self.card)
        dispatch = module.Dispatch(self.device, self.pll_mhz)
        dispatch.settle_us = settle_us
        backend, launched = None, False
        try:
            state_t = _state_area(self.device)
            tag = dispatch.prepare(
                depth,
                host_depth,
                event_depth,
                start,
                deadline,
                kernel_idle,
                state_t.buffer_address(),
            )
            backend = _Backend(self, dispatch, mhz, trace)
            backend.state_t = state_t  # kept while the kernel may use it
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace
            )
            self.log("launch")
            try:
                ttnn.generic_op(backend.io, descriptor)
            except Exception as err:
                raise ProgramError(f"{program.name}: launch failed: {err}") from err
            launched = True
            dispatch.start(str(image))
            self.log("dispatch kernel started")
            state = dispatch.wait_ready(start + 1.0)
        except BaseException:
            dispatch.close()
            if launched:
                backend.read_back()  # the cores leave on their own bounded waits
            raise
        run = Run(backend, (dispatch.grid_x, dispatch.grid_y), program, batch)
        self.run = run
        if state != abi.headers()["KMD_STATE_READY"]:
            run.abort()
            raise StartError(
                f"the dispatch kernel did not report READY (state {state})"
            )
        self.log("ready")
        return run


class _Backend:
    """The card side of a Run: the native port, the device tensors, the readback."""

    def __init__(self, session, dispatch, mhz, trace):
        self.session = session
        self.device = session.device
        self.port = dispatch
        self.mhz = mhz
        self.trace_cap = (trace + 3) // 4 * 4 if trace else 0  # whole 64-byte pages
        self.io = []
        self.ctl = None
        self.ctl_offset = 0
        self.trace_t = None
        self.state_t = None
        self.read = False

    def _control_blocks(self, cores, grid):
        """Zeroed control blocks, one shard per core, padded so that the block can
        start on a 64-byte line: the same L1 address on every core."""
        h = abi.headers()
        words = (h["KM_CTL_BYTES"] + CTL_ALIGN) // 4
        config = ttnn.create_sharded_memory_config(
            (1, words),
            ttnn.CoreGrid(y=grid[1], x=grid[0]),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.ctl = _int32_tensor(self.device, np.zeros((cores, words)), config)
        base = self.ctl.buffer_address()
        addr = (base + CTL_ALIGN - 1) // CTL_ALIGN * CTL_ALIGN
        self.ctl_offset = (addr - base) // 4
        return addr

    def describe(self, program, sources, tag, depth, wait, idle, trace):
        """The program descriptor for every core, with the start-up arguments."""
        h = abi.headers()
        d = self.port
        grid = (d.grid_x, d.grid_y)
        cores = grid[0] * grid[1]
        ctl = self._control_blocks(cores, grid)
        table = np.zeros((1, h["KM_CORES_MAX"]), dtype=np.uint64)
        table[0] = d.core_table()
        cores_t = _int32_tensor(self.device, table)
        self.io = [self.ctl, cores_t]
        if self.trace_cap:
            self.trace_t = _int32_tensor(
                self.device, np.zeros((cores, self.trace_cap * h["kTraceWords"]))
            )
            self.io.append(self.trace_t)
        base = d.lim_base
        groups = abi.column_groups(grid, cores)
        common = {
            "ctl": ctl,
            "cores": cores_t.buffer_address(),
            "disp_xy": d.l2_x | d.l2_y << 16,
            "ack_base": base + h["KMD_LIM_ACKS"],
            "ready_base": base + h["KMD_LIM_READY"],
            "tag": tag,
            "depth": depth,
            "row": grid[0],
            "wait_limit": abi.limit_units(wait, self.mhz),
            "idle_limit": abi.limit_units(idle, self.mhz) if idle else 0,  # 0: none
            "trace": self.trace_t.buffer_address() if self.trace_t else 0,
            "trace_cap": self.trace_cap,
        }
        reader_args, writer_args, compute_args = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        for core in range(cores):
            x, y = core % grid[0], core // grid[0]
            leader, rank, members, slot = groups[core]
            start = abi.start_args(
                common
                | {
                    "me": core,
                    "mailbox": base + h["KMD_LIM_MAILBOX"] if core == 0 else 0,
                    "leader": leader,
                    "rank": rank,
                    "members": members,
                    "ready_slot": slot,
                }
            )
            reader, writer, compute = program.core_args(core)
            reader_args[x][y] = start + reader
            writer_args[x][y] = start + writer
            limits = [common["wait_limit"], common["idle_limit"]]
            compute_args[x][y] = [ctl, core, *limits, *compute]
        all_cores = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*(g - 1 for g in grid))
                )
            ]
        )
        include = [abi.headers().abi.parents[2]]
        defines = [(k, str(v)) for k, v in program.defines.items()]

        def cb(index, fmt, page, pages):
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=index, data_format=fmt, page_size=page
                    )
                ],
            )

        tok = h["kTokenBytes"]
        cbs = [
            cb(b.index, getattr(ttnn, b.format), b.page_bytes, b.pages)
            for b in program.buffers
        ]
        cbs += [
            cb(h["kCbTokenCompute"], ttnn.uint32, tok, 1),
            cb(h["kCbTokenWriter"], ttnn.uint32, tok, 1),
        ]
        if self.trace_cap:
            cbs.append(cb(h["kCbTrace"], ttnn.uint32, self.trace_cap * 16, 1))

        def dm(kernel, args, riscv):
            processor, noc = (
                (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default)
                if riscv == 0
                else (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default)
            )
            return ttnn.KernelDescriptor(
                kernel_source=str(sources[kernel]),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=all_cores,
                compile_time_args=list(program.compile_args.get(kernel, [])),
                defines=defines,
                runtime_args=args,
                config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
                compiler_include_paths=include,
            )

        compute_config = {
            k: getattr(ttnn.MathFidelity, v) if k == "math_fidelity" else v
            for k, v in program.compute.items()
        }
        return ttnn.ProgramDescriptor(
            kernels=[
                dm("reader", reader_args, 0),
                dm("writer", writer_args, 1),
                ttnn.KernelDescriptor(
                    kernel_source=str(sources["compute"]),
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=all_cores,
                    compile_time_args=list(program.compile_args.get("compute", [])),
                    defines=defines,
                    runtime_args=compute_args,
                    config=ttnn.ComputeConfigDescriptor(**compute_config),
                    compiler_include_paths=include,
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )

    def finish(self, timeout):
        """Waits for the kernel to finish, stops it, then closes the X280 runtime."""
        try:
            return self.port.finish(timeout)
        finally:
            self.port.close()
            self.session.log("x280 runtime closed")

    def abort(self):
        cleaned = self.port.close()
        self.session.log(f"x280 runtime closed after a stop (cleaned {cleaned})")
        return cleaned

    def read_back(self):
        """After the X280 runtime is closed: waits for every kernel of the grid to
        return, then reads the state words and the trace."""
        if self.port.is_open:
            raise RuntimeError("close the X280 runtime before reading back")
        t0 = time.monotonic()
        ttnn.synchronize_device(self.device)
        self.session.log(f"grid returned ({time.monotonic() - t0:.1f} s)")
        h = abi.headers()
        rows = _read_uint32(self.ctl)
        start = self.ctl_offset + h["KM_CTL_STATE"] // 4
        state = rows[:, start : start + h["kStateWords"]].astype(np.uint32)
        trace = None
        if self.trace_t is not None:
            words = _read_uint32(self.trace_t).astype(np.uint32)
            trace = words.reshape(words.shape[0], self.trace_cap, h["kTraceWords"])
        self.session.log("state and trace read back")
        return state, trace
