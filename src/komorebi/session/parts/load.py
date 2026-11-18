"""The session's loads: a program onto the card or its islands, the backend's Card,
the kernel buffer's resize."""

import copy
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from komorebi import abi
from komorebi.device import kernel_buffer, mock, native
from komorebi.errors import (
    KernelSizeError,
    L1BudgetError,
    ProgramError,
    StartError,
)
from komorebi.program import several as programs
from komorebi.program import thread as thread_module
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.program.checks import instructions as insn_check
from komorebi.program.description import (
    Program,
    guard_says,
    unguarded,
)
from komorebi.session import kernels as kernel_module
from komorebi.session import process, sizes, tensor_bytes
from komorebi.session import x280 as x280_module
from komorebi.session.backend.backend import _Backend
from komorebi.session.backend.card import Card
from komorebi.session.run import Run

# The X280 clock: 1750 MHz loses loads under full Tensix load. 1350 is the default; a
# runtime without it gets 1000 (its error: "no PLL solution") unless one was asked for.
X280_MHZ = 1350
X280_FALLBACK_MHZ = 1000
NO_PLL_SOLUTION = "no PLL solution for"


class LoadMixin:
    """The session's loads: a program onto the card or its islands, the backend's Card,
    the kernel buffer's resize."""

    # an island's load beside another island's X280 thread: the same work rings'
    # layout, no ring in use (IslandSession.load), so the upload regions agree
    _rings_beside = False

    # the state of the session's other parts this one reads or sets, owned there
    _default_clock: bool  # Session
    _memory: Any  # Session
    _own_device: bool  # Session
    card: int | None  # Session
    device: Any  # Session
    grid_timeout: float  # Session
    kernel_bytes: int | None  # Session
    kernel_followed: bool  # Session
    log: Callable  # Session
    pll_mhz: int  # Session
    slow_code: bool  # Session
    tree: Any  # Session
    _mock_mapped: Callable  # OpenMixin
    _module: Any  # OpenMixin
    channel: Any  # OpenMixin
    stuck: bool  # OpenMixin
    _kept: Any  # MemoryMixin
    _taken: list | None  # MemoryMixin
    l1_held: Callable  # MemoryMixin
    largest_page: int  # MemoryMixin
    page_bounds: int  # MemoryMixin
    transfer_chunk: int | None  # MemoryMixin

    def _init_load(self):
        """Nothing loaded yet."""
        self._last_total = 0  # the kernels' total of the last load built
        self.run = None
        self.island = None  # (graph, i) of the loaded run: Session.load's `island`
        self.link_area = 0  # its link area's L1 address (Session.load's `link`), or 0
        self.link_plan = None  # Session.load's `link`
        self.link_plans = None  # per island LinkPlan of a load over a split card
        self.packet = 0  # an island session's link packet bytes (0: none)
        self.tensor_table = 0  # its tensor table's L1 address (load's `tensors`)
        self._starting = None  # the Dispatch of a load before its Run exists

    def _card(self):
        """What a backend of this session's load reads of it (backend.card.Card): the
        load's values, fixed before the backend is made, and the session's calls."""
        return Card(
            log=self.log,
            module=self._module,
            tree=self.tree,
            device=self.device,
            island=self.island,
            link_area=self.link_area,
            link_plan=self.link_plan,
            link_plans=self.link_plans,
            packet=self.packet,
            tensor_table=self.tensor_table,
            page_bounds=self.page_bounds,
            slow_code=self.slow_code,
            grid_timeout=self.grid_timeout,
            transfer_chunk=self.transfer_chunk,
            channel=self.channel is not None or self._mock_mapped(),
            tensor=self.tensor,
            empty=self.empty,
            write=self.write,
            l1=self.l1,
            l1_rows=self.l1_rows,
            l1_held=self.l1_held,
            l1_taken=self.l1_taken,
            nocs=self.nocs,
            page_bounds_list=self.page_bounds_list,
            largest_page=lambda: self.largest_page,
            hand_nocs=self._hand_nocs,
            resize=self._resize,
            mark_stuck=self._mark_stuck,
        )

    def _mark_stuck(self):
        """The grid did not return after a stop: the runtime closes it at exit."""
        self.stuck = True

    def _open_dispatch(self, module):
        """The runtime's dispatch at the session's clock. A runtime that does not offer
        the default clock gets 1000, logged once; an asked-for clock is an error."""
        if mock.active():  # a mock has no X280
            return mock.Dispatch(self.device)
        try:
            return module.Dispatch(self.device, self.pll_mhz)
        except RuntimeError as error:
            if not (self._default_clock and NO_PLL_SOLUTION in str(error)):
                raise
            self.log(
                f"X280 clock {self.pll_mhz} MHz is not offered by this runtime "
                f"({error}); using {X280_FALLBACK_MHZ} MHz"
            )
        self.pll_mhz = X280_FALLBACK_MHZ
        return module.Dispatch(self.device, self.pll_mhz)

    def load(self, program, *args, **options):
        """Launches `program` (a list or {program: (first, count)}: the runtime's form)
        on every core, then the dispatcher and the X280 thread; returns a Run
        (docs/reference/host.md). A session not given `kernel_bytes` opens its card
        again first when the kernels need another kernel buffer (kernel_buffer)."""
        loading, self._kept.loading = self._kept.loading, True
        try:
            run = self._try_load(program, *args, **options)
        except kernel_buffer.Resize as resize:
            size, why = resize.size, resize.why
        else:
            self._kept.stop()
            self._remember(program)
            return run
        finally:
            self._kept.loading = loading
        kernel_buffer.reopen(self, size, sizes.worker_l1_size(size), why)
        self._kept.loading = True
        try:
            run = self._try_load(program, *args, **options)
        finally:
            self._kept.loading = loading
        self._kept.stop()
        self._remember(program)
        return run

    def _remember(self, program):
        """The kernel buffer a load needed, kept for the script's next run."""
        if self._memory is not None and self._last_total:
            self._memory.loaded(program, self._last_total)

    def _try_load(self, program, *args, **options):
        """_load with the L1 the session held before it (l1_taken) fixed for its
        table: the load's own L1 is counted by its own rows."""
        self._taken = self.l1_taken()
        try:
            return self._load(program, *args, **options)
        finally:
            self._taken = None

    def _log_kernel_buffer(self, program):
        """The built kernels' line: their bytes of the kernel buffer, each kernel's
        text (a load's, a split card's too)."""
        text = ", ".join(f"{name} {size}" for name, size in program.sizes.text.items())
        self.log(
            f"kernel buffer: {program.sizes.total} of {program.sizes.limit} B "
            f"(text: {text})"
        )

    def _resize(self, total, l1_failed=False):
        """Raises kernel_buffer.Resize when this session follows its kernels (no
        `kernel_bytes` given, or one followed; its own card alone) and kernels of
        `total` bytes need another buffer: a larger one, or a smaller one when the L1
        table failed. Returns why it does not follow them, or None."""
        size, buffer = kernel_buffer.need(total), self.l1()[0]
        grow = size > buffer
        if not (grow or (l1_failed and size < buffer)):
            return None
        refused = (
            "the session was given kernel_bytes"
            if self.kernel_bytes is not None and not self.kernel_followed
            else (
                "it did not open its card"
                if not self._own_device or self.island is not None
                else (
                    f"{size} B is over the most, {sizes.KERNEL_BUFFER_MAX} B"
                    if size > sizes.KERNEL_BUFFER_MAX
                    else self._kept.refused()
                )
            )
        )
        if refused is not None:
            if self._memory is not None and grow:  # the next run opens right
                self._memory.wanted(total)
                refused += " (remembered: the script's next run opens at it)"
            return f"{refused}; Session(kernel_bytes={size}) holds them"
        why = (
            f"the kernels take {total:,} B"
            if grow
            else f"the L1 table does not fit and the kernels take {total:,} B"
        )
        raise kernel_buffer.Resize(size, why)

    def _load(
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
        settle_microseconds=1.0,
        image=None,
        thread="card",
        island=None,
        link=None,
        tensors=0,
        recover=True,
        reserve=added.RESERVE,
        buffers=None,
    ):
        """Session.load's one try on the card as it is opened."""
        arguments = {k: v for k, v in locals().items() if k != "self"}
        fits = getattr(program, "fits_kernels", None)  # a merged class's: a load again
        if fits is not None and not program.fitting:  # with a handler as an overlay
            return fits(self, arguments)
        if reserve < 0:
            raise ValueError(f"reserve={reserve}: bytes, 0 or more")
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        self.island = kernel_module._island(island, self.grid)
        if link is not None and self.island is None:
            raise ValueError("link= is an island's (island= too)")
        self.link_plan = link
        self.link_area = link.area if link is not None else 0
        self.tensor_table = tensors
        self._hand_nocs(program)
        placed = None
        if isinstance(program, programs.Plan):  # an island's part of a flat plan
            placed = program
            program = placed.programs[0]
        elif not isinstance(program, Program) or program.form() == "program":
            grid = self.grid
            placed = programs.plan(program, grid[0] * grid[1])
            program = placed.programs[0]
        owner = placed.threads_owner() if placed is not None else program
        if thread not in ("card", "host"):
            raise ValueError(f"thread={thread!r}: 'card' or 'host'")
        works = [owner.threads[cluster] for cluster in sorted(owner.threads)]
        owner.check_threads()
        if owner.x280_ports is not None and (not works or thread != "card"):
            raise ValueError(
                f"{owner.name}: X280 ports need its X280 threads on the card"
            )
        for work in works:
            if thread == "host" and work.host_fallback is None:
                raise ValueError(
                    f"{owner.name}: the thread on cluster {work.cluster} has no "
                    "host_fallback"
                )
        several = (
            placed.programs if placed is not None and len(placed.programs) > 1 else ()
        )
        for work in works:  # a class's thread: its header with the load's programs
            rewritten = getattr(work, "programs_header", None)
            if rewritten is not None and thread == "card":
                rewritten(several)
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatcher image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        on_card = bool(works) and thread == "card"
        images = (
            {
                work.cluster: native.thread_image(work, self.tree, several)
                for work in works
            }
            if on_card
            else {}
        )
        ranks = x280_module._x280_ranks(program, owner, placed, on_card)
        depth = depth or program.depth
        sources = program.sources()  # before anything opens: a missing file raises
        if placed is not None:
            sources = programs.runtime_kernels()
        mhz = mock.MHZ if mock.active() else process.aiclk_mhz(self.card)
        # the regions go to the card before the grid starts: a tensor write waits
        # for every core to be idle, which a persistent grid never is
        regions = x280_module._bank_regions(self, owner, works)
        dispatch = self._open_dispatch(module)
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        process._hook_exit()
        process._LIVE.add(self)
        self._starting = dispatch
        dispatch.settle_microseconds = settle_microseconds
        backend, launched = None, False
        try:
            if not dispatch.idle:
                raise StartError("X280 cluster 0 is not idle: no image can start")
            state_tensor = tensor_bytes._state_area(self)
            rings = {
                work.cluster: x280_module._work_ring(dispatch, work)
                for work in works
                if on_card
            }
            ends = {cluster: end for cluster, (_, end) in rings.items()}
            tag = dispatch.prepare(
                depth,
                host_depth,
                event_depth,
                start,
                deadline,
                kernel_idle,
                state_tensor.address,
                **thread_module.kernel_work_args(ends, self._rings_beside),
                messages_handler=(
                    program.messages.dispatcher_word("KM_DELIVER" in program.defines)
                    if program.messages is not None
                    else 0
                ),
                programs=(
                    kernel_module._held_fields(
                        placed,
                        kernel_module._link_kernels(
                            self.island,
                            program,
                            None,
                            dispatch.grid_x * dispatch.grid_y,
                        )[1],
                    )
                    if placed and placed.switch
                    else []
                ),
                island=kernel_module._island_word(self.island),
                link=list(link.line(dispatch)) if link is not None else [],
                channel_region=self._channel_region(),
            )
            kernel_module.check_bases(kernel_module.lim_bases(dispatch.lim_base))
            # an island's ports serve its ranks, not its link cores
            served, _ = kernel_module._link_kernels(
                self.island, program, None, dispatch.grid_x * dispatch.grid_y
            )
            plan, port_lines = x280_module._port_lines(owner, dispatch, rings, served)
            backend = _Backend(self._card(), dispatch, mhz, trace)
            backend.state_tensor = state_tensor  # kept while the kernel may use it
            backend.reserve = reserve
            backend.added_buffers = added_program.buffers_given(buffers)
            if ranks is not None:
                ranks = x280_module._x280_words(
                    self, program, dispatch, rings, regions, plan
                )
            descriptor = backend.describe(
                program, sources, tag, depth, wait, idle, trace, port_lines, placed,
                ranks,
            )  # fmt: skip
            started, built_at = time.perf_counter(), time.time()
            try:
                elves, *build = descriptor.build()
            except Exception as error:
                raise ProgramError(f"{program.name}: build failed: {error}") from error
            program.sizes = kernel_module.kernel_sizes(build)
            self._last_total = program.sizes.total
            for other in placed.programs if placed is not None else ():
                other.sizes = program.sizes
            self._log_kernel_buffer(program)
            if build[-1]:
                self._log_built(descriptor, built_at, program)
                refused = self._resize(program.sizes.total or sizes.KERNEL_BUFFER_MAX)
                raise KernelSizeError(
                    program.name, program.sizes, sizes.KERNEL_BUFFER_MAX, refused
                )
            if insn_check.enabled():
                insn_check.check_kernels(
                    program.name,
                    elves,
                    native.metal_tree(self.tree),
                    native.cache_directory() / "insn-check",
                )
                self.log(
                    f"kernels built and checked ({time.perf_counter() - started:.2f} s)"
                )
            if placed is not None:
                backend.load_programs(placed, descriptor)
            else:
                backend.load_overlays(program, descriptor)
            self._log_built(descriptor, built_at, program)
            if mock.active():  # built: a mock run stands in for the launch
                self._starting = None
                host = self._made_run(backend, dispatch, program, batch, placed, owner)
                self._start_threads(
                    host, works, owner, program, dispatch, backend, images, regions,
                    rings, on_card, plan, port_lines, ranks, wait, mhz, link,
                )  # fmt: skip
                self.run = mock.Run(counted=self.island is None, host=host)
                return self.run
            if backend.apart_links:  # after every other write: a transfer's buffer
                link.clear(self)  # starts at the link cores' buffers' base too
            self.log("launch")
            # From here the grid may run: an interrupt is delivered when the launch
            # call returns, after the kernels started.
            launched = True
            try:
                descriptor.run()
            except Exception as error:
                raise ProgramError(f"{program.name}: launch failed: {error}") from error
            backend.launch = descriptor  # its program: a core's kernels reloaded
            dispatch.start(str(image))
            self.log("dispatcher started")
            state = dispatch.wait_ready(start + 1.0)
        except BaseException:
            try:
                if not dispatch.close():
                    self.log(f"x280 runtime not restored: {dispatch.close_failed}")
            except BaseException:
                if launched:  # the runtime stays open: the exit hooks close it again
                    backend.stop_grid()
                raise
            self._starting = None
            if launched:
                backend.read_back(stop_grid=True)  # the kernel never stopped it
            raise
        run = self._made_run(backend, dispatch, program, batch, placed, owner)
        self.run = run
        self._starting = None
        if state != abi.headers()["KMD_STATE_READY"]:
            run._fail_start(state)
        if placed is not None and placed.switch:
            kernel_module._start_loads(run, placed, part=self.island is not None)
        if recover and not (placed is not None and placed.switch):
            # an island's link cores keep their held connection: no reset list
            exclude = ()
            if self.island is not None:
                graph, here = self.island
                exclude = set(range(run.cores)) - set(graph.rank_cores[here])
            if run.keep_resets(exclude):
                self.log(
                    f"reset lists kept: {len(run.recovery['lists'])}, "
                    f"{run.recovery['bytes']} B in "
                    f"{run.recovery['seconds'] * 1e3:.1f} ms"
                )
            else:
                self.log(f"no reset list: {run.recovery_refused}")
        elif recover:
            run.recovery_refused = "a run whose groups switch program"
        if unguarded(run.programs):
            self.log(guard_says(unguarded(run.programs), recover))
        self.log("ready")
        self._start_threads(
            run, works, owner, program, dispatch, backend, images, regions, rings,
            on_card, plan, port_lines, ranks, wait, mhz, link,
        )  # fmt: skip
        return run

    def _start_threads(
        self, run, works, owner, program, dispatch, backend, images, regions, rings,
        on_card, plan, port_lines, ranks, wait, mhz, link,
    ):  # fmt: skip
        """The load's X280 threads on `run` (run.threads, run.thread), started: on the
        card after the launch, under a prebuild on the mock's dispatch (its Run's
        threads are the card's, their port the mock's)."""
        if works:
            shared = {
                name: x280_module._host_array(value)
                for name, value in owner.shared.items()
            }
            peers = x280_module._peers(self, dispatch, images) if on_card else {}
            switch = getattr(backend, "switch", None)
            records = thread_module.load_records(switch) if on_card else None
            if records is not None:  # a thread may load: reason 13 is the judge
                run.check_programs = False
            try:
                for work in works:
                    cluster = work.cluster
                    view = regions[work.bank].view(work)
                    thread_run = thread_module.ThreadRun(
                        work, dispatch, view, images.get(cluster), on_card, view.base,
                        view.check, run, shared,
                    )  # fmt: skip
                    thread_run.ring = rings[cluster][0] if on_card else None
                    thread_run.peers = peers.get(cluster)
                    thread_run.records = records
                    if self.island is not None:
                        graph, here = self.island
                        thread_run.island = here | len(graph.islands) << 16
                    if backend.transfer is not None:
                        thread_run.buffer = backend.transfer.address
                    if plan is not None:
                        thread_run.ports = thread_module.Ports(
                            backend.control_addr,
                            len(port_lines.cores),
                            port_lines.cores,
                            plan,
                            owner.x280_ports["noc"],
                        )
                        thread_run.write_ports()
                    if ranks is not None and cluster == program.messages.x280_cluster:
                        thread_run.ranks = x280_module._x280_line(
                            program, dispatch, backend, ranks, wait, mhz,
                            link_area=getattr(link, "area", 0),
                        )  # fmt: skip
                        thread_run.write_ranks()
                    run.threads[cluster] = thread_run
                for (
                    thread_run
                ) in run.threads.values():  # every ports area is written first
                    thread_run.start()
            except BaseException:
                run.abort()
                raise
            run.thread = run.threads[works[0].cluster]
            where = "on the card" if on_card else "on the host"
            self.log(f"threads {list(run.threads)} {where}")

    def _made_run(self, backend, dispatch, program, batch, placed, owner):
        """The load's Run over `backend` (a mock's too: its run.chain builds and
        refuses as the card's)."""
        run = Run(
            backend,
            (dispatch.grid_x, dispatch.grid_y),
            program,
            batch,
            programs=placed.programs if placed is not None else None,
            holds=placed.holds() if placed is not None else None,
            thread_program=owner,
            switch=getattr(backend, "switch", None),
            ranks=self.island[0].ranks if self.island else None,
            island_first=self.island[0].first(self.island[1]) if self.island else 0,
            island_ranks=self.island[0].count(self.island[1]) if self.island else None,
            thread_first=(
                self.island[0].thread_numbers(self.island[1]).start
                if self.island
                else 1
            ),
        )
        run.dram = self._dram  # its stamps' read-out checked against these
        return run

    def _island_runs(self, backends, graph, rows, program, batch):
        """A split card's Run an island, over its backend (rows: each island's)."""
        runs = []
        for island, backend in enumerate(backends):
            run = Run(
                backend,
                self.grid,
                program,
                batch,
                ranks=graph.ranks,
                island_first=graph.first(island),
                island_ranks=graph.count(island),
                island_cores=(rows[island][0] * self.grid[0], graph.count(island)),
                island_span=rows[island][1] * self.grid[0],
            )
            run.recovery_refused = "a card split into islands"
            run.dram = self._dram
            runs.append(run)
        return runs

    def load_islands(
        self,
        program,
        graph,
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
        settle_microseconds=1.0,
        image=None,
        links=None,
        tensors=0,
    ):
        """This card split into the islands of `graph` (whole rows, ranks then link
        cores): `program` launched once, island i's dispatcher on X280 cluster i,
        `links` each island's LinkPlan or None, `tensors` the tensor table's address.
        A Run an island over flat ranks; Session.load's options, no X280 threads."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if not isinstance(program, Program) or program.form() == "program":
            raise ValueError("a card split into islands loads one Program")
        if (
            program.threads
            or program.x280_ports
            or (program.messages is not None and program.messages.x280_ranks)
        ):
            raise ValueError(
                f"{program.name}: X280 threads, ports and ranks need the work "
                "clusters, which a split card's dispatchers take"
            )
        rows = sizes._split_rows(graph, self.grid)
        self.link_plans, self.link_plan, self.link_area = links, None, 0
        self.tensor_table = tensors
        image = Path(image) if image else native.image_path(self.tree)
        if not image.is_file() or image.stat().st_size == 0:
            raise StartError(f"dispatcher image {image}: missing or empty")
        module = self._module = native.load_module(self.tree)
        depth = depth or program.depth
        sources = program.sources()
        mhz = mock.MHZ if mock.active() else process.aiclk_mhz(self.card)
        dispatches = kernel_module._Dispatches([self._open_dispatch(module)])
        self.log(f"X280 clock set: {self.pll_mhz} MHz")
        process._hook_exit()
        process._LIVE.add(self)
        self._starting = dispatches
        backends, launched = [], False
        try:
            for cluster in range(1, len(rows)):
                dispatches.append(
                    mock.Dispatch(self.device)
                    if mock.active()
                    else module.Dispatch(dispatches[0], cluster)
                )
            states, tags = [], []
            for island, (dispatch, (first_row, count)) in enumerate(
                zip(dispatches, rows)
            ):
                dispatch.settle_microseconds = settle_microseconds
                if not dispatch.idle:
                    raise StartError(f"X280 cluster {island} is not idle")
                states.append(tensor_bytes._state_area(self))
                base = first_row * self.grid[0]
                plan = links[island] if links else None
                line = b""
                if plan:
                    line = plan.line(dispatch, dispatches[plan.far], dispatches)
                tags.append(
                    dispatch.prepare(
                        depth,
                        host_depth,
                        event_depth,
                        start,
                        deadline,
                        kernel_idle,
                        states[-1].address,
                        messages_handler=(
                            program.messages.dispatcher_word(
                                "KM_DELIVER" in program.defines
                            )
                            if program.messages is not None
                            else 0
                        ),
                        island=kernel_module._island_word((graph, island)) | base << 32,
                        link=list(line),
                        first_row=first_row,
                        rows=count,
                        channel_region=self._channel_region(),
                    )  # fmt: skip
                )
                kernel_module.check_bases(kernel_module.lim_bases(dispatch.lim_base))
            if len({dispatch.lim_base for dispatch in dispatches}) != 1:
                raise StartError(  # a core's words name one LIM address on every one
                    "the islands' dispatchers differ in LIM base: "
                    f"{[hex(dispatch.lim_base) for dispatch in dispatches]}"
                )
            if len({dispatch.upload_bytes for dispatch in dispatches}) != 1:
                raise StartError(  # the link area after it: one place on every one
                    "the islands' dispatchers differ in upload region: "
                    f"{[dispatch.upload_bytes for dispatch in dispatches]}"
                )
            for ring in (True, False):  # the slots fit only without the put rings
                first = _Backend(self._card(), dispatches[0], mhz, trace)
                first.split = (graph, list(dispatches), tags, rows, set())
                first.state_tensor = states[0]
                first.link_ring = ring
                descriptor = first.describe(
                    program, sources, tags[0], depth, wait, idle, trace
                )
                built_at = time.time()
                try:
                    elves, *build = descriptor.build()
                except Exception as error:
                    raise ProgramError(
                        f"{program.name}: build failed: {error}"
                    ) from error
                program.sizes = kernel_module.kernel_sizes(build)
                self._last_total = program.sizes.total
                self._log_kernel_buffer(program)
                if build[-1]:
                    self._log_built(descriptor, built_at, program)
                    total = program.sizes.total or sizes.KERNEL_BUFFER_MAX
                    refused = self._resize(total)  # a followed size: Resize
                    raise KernelSizeError(
                        program.name, program.sizes, sizes.KERNEL_BUFFER_MAX, refused
                    )
                if insn_check.enabled():
                    insn_check.check_kernels(
                        program.name,
                        elves,
                        native.metal_tree(self.tree),
                        native.cache_directory() / "insn-check",
                    )
                try:
                    first.load_overlays(program, descriptor)
                    break
                except L1BudgetError:
                    if not ring or not kernel_module.rings_dropped(first):
                        raise
                    first.release_l1()
                    self.log(
                        "the link cores have no put ring: a km::put_bytes batch to "
                        "them is refused"
                    )
            self._log_built(descriptor, built_at, program)
            for island, dispatch in enumerate(dispatches):
                backend = first if island == 0 else copy.copy(first)
                backend.port, backend.state_tensor = dispatch, states[island]
                backend.rows = rows[island]
                backends.append(backend)
            if mock.active():  # built: a mock run an island for the launch
                self._starting = None
                hosts = self._island_runs(backends, graph, rows, program, batch)
                runs = [mock.Run(counted=False, host=host) for host in hosts]
                self.run, self.island = runs[0], (graph, 0)
                return runs
            self.log(f"launch: one grid, {len(rows)} islands")
            launched = True
            try:
                descriptor.run()
            except Exception as error:
                raise ProgramError(f"{program.name}: launch failed: {error}") from error
            for island, (backend, dispatch) in enumerate(zip(backends, dispatches)):
                backend.launch = descriptor
                dispatch.start(str(image))
                self.log(
                    f"island {island}: dispatcher started on X280 cluster {island}"
                )
            ready = [dispatch.wait_ready(start + 1.0) for dispatch in dispatches]
        except BaseException:
            if not dispatches.close():
                self.log(f"x280 runtime not restored: {dispatches.close_failed}")
            self._starting = None
            if launched:
                for backend in backends:  # the kernels never stopped the grid
                    backend.read_back(stop_grid=True)
            raise
        runs = self._island_runs(backends, graph, rows, program, batch)
        self.run, self.island = runs[0], (graph, 0)
        self._starting = None
        for run, state in zip(runs, ready):
            if state != abi.headers()["KMD_STATE_READY"]:
                for other in runs:
                    if other is not run:
                        other.abort()
                run._fail_start(state)
        self.log("ready: every island")
        return runs
