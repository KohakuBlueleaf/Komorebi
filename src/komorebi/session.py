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

from komorebi import abi, dram, native
from komorebi import thread as kt
from komorebi.errors import GridTimeout, ProgramError, StartError
from komorebi.run import Run

CTL_ALIGN = 64  # the dispatch kernel stores whole 64-byte lines into the control block
IPI_ACK_DDR = 0x470020  # the X280 runtime's IPI ack word in the cluster's DRAM bank
STATE_PAGES = 32  # pages of KMD_STATE_BYTES, interleaved: several in every DRAM bank
NOC_ALIGN = 16  # a NoC write's source and destination share their low 4 address bits
DRAM_BANKS = 8  # interleaved page i lies in bank i % DRAM_BANKS
CBS = 64  # circular buffers of a Tensix core: one unpack-to-DST mode each


def compute_config(options):
    """The compute kernel's ttnn.ComputeConfigDescriptor from Program.compute: field
    values as given, except `math_fidelity` (a ttnn.MathFidelity name) and
    `unpack_to_dest_mode` (a ttnn.UnpackToDestMode name for every circular buffer, or
    {cb: name} with the others Default)."""
    options = dict(options)
    unpack = options.pop("unpack_to_dest_mode", None)
    if "math_fidelity" in options:
        options["math_fidelity"] = getattr(ttnn.MathFidelity, options["math_fidelity"])
    config = ttnn.ComputeConfigDescriptor(**options)
    if unpack is not None:
        by_cb = (
            unpack if isinstance(unpack, dict) else dict.fromkeys(range(CBS), unpack)
        )
        modes = [ttnn.UnpackToDestMode.Default] * CBS
        for cb, name in by_cb.items():
            modes[cb] = getattr(ttnn.UnpackToDestMode, name)
        config.unpack_to_dest_mode = modes
    return config


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


def lim_bases(lim_base, h=None):
    """The X280 local-memory addresses the kernels are given or derive from the
    runtime's base: the base, every KMD_LIM_* offset from it, the ack stride."""
    h = h or abi.headers()
    bases = {"lim_base": lim_base, "KM_ACK_STRIDE": h["KM_ACK_STRIDE"]}
    for name, offset in h.constants.items():
        if name.startswith("KMD_LIM_"):
            bases[name] = lim_base + offset
    return bases


def check_bases(bases):
    """Refuses bases that are not multiples of NOC_ALIGN: the runtime's small writes
    go slot to slot and rely on them."""
    bad = {k: v for k, v in bases.items() if v % NOC_ALIGN}
    if bad:
        named = ", ".join(f"{k} {v:#x}" for k, v in bad.items())
        raise StartError(f"not a multiple of {NOC_ALIGN} bytes: {named}")


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


def _dram_interleaved(tensor):
    config = tensor.memory_config()
    return (
        config.buffer_type == ttnn.BufferType.DRAM
        and config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    )


def read_tensors(device, module, tensors):
    """The raw pages of interleaved DRAM tensors as [pages, page_bytes] uint8 each:
    one DRAM barrier, one read per bank and tensor, no wait for idle cores."""
    banks = module.dram_banks(device)
    spans, layout = [], []
    for t in tensors:
        if not _dram_interleaved(t):
            raise ValueError("only interleaved DRAM tensors are read by banks")
        page, aligned, pages = (
            t.buffer_page_size(),
            t.buffer_aligned_page_size(),
            t.buffer_num_pages(),
        )
        mine = dram.bank_spans(t.buffer_address(), aligned, pages, banks)
        layout.append((len(spans), len(mine), page, aligned, pages))
        spans += mine
    chunks = module.read_dram(device, spans)
    return [
        dram.pages_in_order(chunks[at : at + n], page, aligned, pages, banks)
        for at, n, page, aligned, pages in layout
    ]


def _read_uint32(tensor, device=None, module=None):
    """A row-major 32-bit tensor's words as int64: by banks when it is in DRAM and a
    module is given, else through ttnn (which waits for idle cores)."""
    if module is not None and _dram_interleaved(tensor):
        (pages,) = read_tensors(device, module, [tensor])
        words = pages.reshape(-1).view(np.uint32).reshape(tuple(tensor.shape))
        return words.astype(np.int64)
    return ttnn.to_torch(tensor).numpy().astype(np.int64) & abi.M32


class Session:
    """One card: `card` selects it, `device` uses an open one; `grid_timeout` bounds
    every wait for the grid's kernels. As a context manager it stops a loaded run at
    the exit (through the runtime on an exception) and closes the device it opened.
    """

    def __init__(
        self,
        card=None,
        device=None,
        pll_mhz=1750,
        tree=None,
        log=None,
        grid_timeout=10.0,
    ):
        self.card = card
        self.device = device
        self._own_device = device is None
        self.pll_mhz = pll_mhz
        self.tree = tree
        self.log = log or (lambda msg: None)
        self.grid_timeout = grid_timeout
        self.run = None
        self.stuck = False  # the grid did not return: the device stays open
        self._module = None

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
        if self.stuck:
            self.log("device left open: the grid did not return")
            return
        if self.device is not None and self._own_device:
            ttnn.close_mesh_device(self.device)
            self.device = None

    def read(self, *tensors):
        """The raw pages of interleaved DRAM tensors ([pages, page_bytes] uint8 each),
        under one DRAM barrier. It does not wait for idle cores, so it may run while a
        program runs: read only what no kernel writes any more (after its event)."""
        self._module = self._module or native.load_module(self.tree)
        return read_tensors(self.device, self._module, tensors)

    def read_bf16(self, *tensors):
        """bfloat16 tile tensors of rank 2 as float32 arrays, read as read() does."""
        out = []
        for t, pages in zip(tensors, self.read(*tensors)):
            if t.layout != ttnn.TILE_LAYOUT or t.dtype != ttnn.bfloat16:
                raise ValueError("read_bf16 takes bfloat16 tile tensors")
            rows, cols = tuple(t.padded_shape)
            out.append(dram.untilize_bf16(pages, rows, cols))
        return out

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
        thread="card",
    ):
        """Launches `program` on every core, then the dispatch kernel, then the
        program's X280 thread (`thread="card"`, or "host" for its host fallback);
        returns a Run. The arguments are described in docs/reference/host.md.
        """
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if thread not in ("card", "host"):
            raise ValueError(f"thread={thread!r}: 'card' or 'host'")
        work = next(iter(program.threads.values()), None)
        if work is not None and thread == "host" and work.host_fallback is None:
            raise ValueError(f"{program.name}: the thread has no host_fallback")
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatch kernel image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        on_card = work is not None and thread == "card"
        work_image = native.thread_image(work, self.tree) if on_card else None
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        mhz = aiclk_mhz(self.card)
        # the regions go to the card before the grid starts: a tensor write waits
        # for every core to be idle, which a resident grid never is
        regions = _Regions(self, work) if work is not None else None
        dispatch = module.Dispatch(self.device, self.pll_mhz)
        dispatch.settle_us = settle_us
        backend, launched = None, False
        try:
            if not dispatch.idle:
                raise StartError("X280 cluster 0 is not idle: no image can start")
            state_t = _state_area(self.device)
            ring, work_args = _work_ring(dispatch, work) if on_card else (None, ())
            tag = dispatch.prepare(
                depth,
                host_depth,
                event_depth,
                start,
                deadline,
                kernel_idle,
                state_t.buffer_address(),
                *work_args,
            )
            check_bases(lim_bases(dispatch.lim_base))
            backend = _Backend(self, dispatch, mhz, trace)
            backend.state_t = state_t  # kept while the kernel may use it
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace
            )
            self.log("launch")
            # From here the grid may run: an interrupt is delivered when the launch
            # call returns, after the kernels started.
            launched = True
            try:
                ttnn.generic_op(backend.io, descriptor)
            except Exception as err:
                raise ProgramError(f"{program.name}: launch failed: {err}") from err
            dispatch.start(str(image))
            self.log("dispatch kernel started")
            state = dispatch.wait_ready(start + 1.0)
        except BaseException:
            dispatch.close()
            if launched:
                backend.read_back(stop_grid=True)  # the kernel never stopped it
            raise
        run = Run(backend, (dispatch.grid_x, dispatch.grid_y), program, batch)
        self.run = run
        if state != abi.headers()["KMD_STATE_READY"]:
            run._fail_start(state)
        self.log("ready")
        if work is not None:
            try:
                run.thread = self._start_thread(
                    dispatch, work, regions, work_image, on_card, run, ring
                )
            except BaseException:
                run.abort()
                raise
            self.log("thread started" if on_card else "thread on the host")
        return run

    def _start_thread(self, dispatch, work, regions, image, on_card, run, ring):
        """Starts the thread on the card with its command ring, or readies its host
        fallback (which issues through `run`)."""
        thread = kt.ThreadRun(
            work, dispatch, regions, image, on_card, regions.base, regions.check, run
        )
        thread.ring = ring
        thread.start()
        return thread


def _work_ring(dispatch, work):
    """The thread's ring (RingPlacement) and the kernel's work_* arguments."""
    info = dispatch.work_info(work.cluster)
    return kt.work_ring(
        dispatch.lim_base,
        (dispatch.l2_x, dispatch.l2_y),
        info["lim_base"],
        (info["virtual_x"], info["virtual_y"]),
    )


def _pages(data, pages):
    """`data` (uint8) zero-padded to `pages` region pages, as [pages, words] uint32."""
    out = np.zeros(pages * kt.PAGE, dtype=np.uint8)
    out[: data.size] = data
    return out.view(np.uint32).reshape(pages, -1)


class _Regions:
    """A thread's regions in one DRAM tensor whose pages of the work cluster's bank
    hold them (page 0: the check line). Written before the grid starts; read(name)
    only once the grid has stopped (a tensor read waits for idle cores)."""

    def __init__(self, session, work):
        self.session = session
        device = session.device
        self.check = kt.new_check_value()
        self.bank = kt.BANK_OF_CLUSTER[work.cluster]
        placed, pages = kt.place(work)
        self.placed = {p.name: p for p in placed}
        self.readable = False  # set by the run once the grid has stopped
        rows = np.zeros((DRAM_BANKS * pages, kt.PAGE // 4), dtype=np.uint32)
        mine = rows[self.bank :: DRAM_BANKS]
        mine[0, :2] = [self.check & abi.M32, self.check >> 32]
        for name, p in self.placed.items():
            data = work.initial(name)
            if data is not None:
                mine[p.page : p.page + p.pages] = _pages(data, p.pages)
        self.tensor = _int32_tensor(device, rows)
        self.base = self.tensor.buffer_address()
        if self.base <= kt.IPI_ACK_DDR < self.base + pages * kt.PAGE:
            raise StartError(
                f"the thread's regions at {self.base:#x} cover the IPI word"
            )

    def address(self, name):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.bank, self.base + self.placed[name].page * kt.PAGE

    def read(self, name):
        """The region's bytes, as uint8."""
        if not self.readable:
            raise RuntimeError("regions are read after run.stop()")
        p = self.placed[name]
        s = self.session
        rows = _read_uint32(self.tensor, s.device, s._module).astype(np.uint32)
        mine = rows[self.bank :: DRAM_BANKS][p.page : p.page + p.pages]
        return mine.reshape(-1).view(np.uint8)[: p.bytes].copy()


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
        self.ctl_addr = 0
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
        self.ctl_addr = (base + CTL_ALIGN - 1) // CTL_ALIGN * CTL_ALIGN
        self.ctl_offset = (self.ctl_addr - base) // 4
        check_bases({"control block": self.ctl_addr})
        return self.ctl_addr

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
            "workers": cores,
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
                    config=compute_config(program.compute),
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

    def read_back(self, stop_grid=False):
        """After the X280 runtime is closed: waits (bounded) for the grid's kernels,
        then reads the control blocks ([cores, KM_CTL_BYTES / 4] uint32) and the trace.
        `stop_grid`: the dispatch kernel was not there to stop the grid, so the host
        first writes the stop word into every core through the device API."""
        if self.port.is_open:
            raise RuntimeError("close the X280 runtime before reading back")
        h = abi.headers()
        module, gx, gy = self.session._module, self.port.grid_x, self.port.grid_y
        if stop_grid:
            for core in range(gx * gy):
                at = self.ctl_addr + h["KM_CTL_STOP"]
                module.write_l1(self.device, core % gx, core // gx, at, [1])
            self.session.log("stop word written by the host")
        t0 = time.monotonic()
        bound = self.session.grid_timeout
        if not module.wait_grid(self.device, gx, gy, bound):
            self.session.stuck = True
            raise GridTimeout(
                f"the grid's kernels did not return within {bound} s after the stop; "
                "the device is left open and the card may need a reset"
            )
        ttnn.synchronize_device(self.device)  # returns at once: every core is done
        self.session.log(f"grid returned ({time.monotonic() - t0:.1f} s)")
        rows = _read_uint32(self.ctl)
        start = self.ctl_offset
        blocks = rows[:, start : start + h["KM_CTL_BYTES"] // 4].astype(np.uint32)
        trace = None
        if self.trace_t is not None:
            words = _read_uint32(self.trace_t, self.device, module).astype(np.uint32)
            trace = words.reshape(words.shape[0], self.trace_cap, h["kTraceWords"])
        self.session.log("state and trace read back")
        return blocks, trace
