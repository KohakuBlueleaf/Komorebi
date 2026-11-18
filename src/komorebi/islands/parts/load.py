"""The island session's load: one program over its islands (a split card, cards)."""

import inspect
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from komorebi import abi
from komorebi.device import kernel_buffer, mock, native, remembered
from komorebi.errors import (
    KernelSizeError,
    KomorebiError,
    L1BudgetError,
)
from komorebi.islands import records as island_records
from komorebi.islands import run as island_run
from komorebi.islands.parts.link_plan import (
    CROSS_PAGE,
    ISLANDS_MAX,
    KERNEL_BYTES_SERVICES,
    ROUND_ISLANDS_MAX,
)
from komorebi.program import several as programs_module
from komorebi.program.description import Program
from komorebi.session import sizes
from komorebi.session.parts import load as card_load

# Session.load's options in their order, for options given by position
LOAD_OPTIONS = tuple(inspect.signature(card_load.LoadMixin._load).parameters)[2:]

# Two cards' kernel buffer: the largest build, a dynamic program with a message service
# a link (about 159,400 B), in steps of 16 KiB. kernel_bytes= overrides it.
KERNEL_BYTES = 160 * 1024


def put_words(row, h):
    """A counts row's put block words (crossings(puts=True)): the queue's tail, each
    page CPU's lines taken, the lines sent across, confirmed, bad; pending, last."""
    put, queue = h["KM_COUNTS_PUT"] // 4, h["KM_COUNTS_QUEUE"] // 4

    def at(offset):
        return int(row[put + (h[offset] - h["KM_PUT_TAIL"]) // 4])

    return {
        "put_tail": at("KM_PUT_TAIL"),
        "put_taken": tuple(
            int(row[put + (h["KM_PUT_TAKEN"] - h["KM_PUT_TAIL"]) // 4 + j])
            for j in range(h["KM_PAGE_CPUS"])
        ),
        "put_done": at("KM_PUT_DONE"),
        "put_confirmed": at("KM_PUT_CONFIRMED"),
        "put_bad": at("KM_PUT_BAD"),
        "queue_pending": int(row[queue + h["KM_QUEUE_PENDING_COUNT"]]),
        "queue_last": int(row[queue + h["KM_QUEUE_LAST"]]),
    }


def check_link_queues(places):
    """Refuses islands whose link cores' put queues (offset from their buffers' base,
    chunk bytes) differ: a sender writes its lines' arrival at its own queue's place."""
    seen = sorted({place for place in places if place is not None})
    if len(seen) > 1:
        raise KomorebiError(
            "the islands' link cores put queues at different places (offset, chunk "
            f"bytes): {seen}: a put line across would never be confirmed"
        )


class LoadMixin:
    """The island session's load: one program over its islands (a split card, cards)."""

    # the state of the session's other parts this one reads or sets, owned there
    ASKS_BY: str  # IslandSession
    _memory: Any  # IslandSession
    log: Callable  # IslandSession
    message_links: bool  # IslandSession
    _regions: list  # MeshMixin
    card_dram: bool  # MeshMixin
    cards: tuple  # MeshMixin
    graph: Any  # MeshMixin
    kernel_bytes: int | None  # MeshMixin
    link_cores: tuple  # MeshMixin
    links: int  # MeshMixin
    run: Any  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin
    _counts: list  # LinkPlanMixin
    _counts_puts: list  # LinkPlanMixin

    def _crossings(self, timeout=10.0, puts=False):
        """Crossings read now: a COUNTS command to every link core of each island (each
        CPU writes its row into the island's counts tensor), waited for together;
        `puts`: each row also holds its link area's put block words (PUT_WORDS)."""
        h = abi.headers()
        tensors = self._counts_puts if puts else self._counts
        page = h["KM_COUNTS_PUT_PAGE"] if puts else CROSS_PAGE
        waits = []
        for island in range(len(self.graph.islands)):
            words = [0] * 10
            words[h["KM_PAGES_TO"]] = tensors[island].address
            words[h["KM_PAGES_PAGE"]] = page
            words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_COUNTS"] << 24
            part = self.run.island(island)
            handle = part._queue_command(
                h["KM_HANDLER_PAGES"], self.graph.link_group(island), words
            )
            part.own_commands[handle] = "the links' counts for crossing_line"
            waits.append((island, handle))
        self.wait_together(waits, timeout)
        fields = [
            h[f"KM_CROSS_{name.upper()}"]
            for name in island_records.Crossings.FIELDS[:-1]
        ]
        rows = {}
        for island in range(len(self.graph.islands)):
            (raw,) = self.sessions[island].read(tensors[island])
            words = raw.view("<u4").astype(np.int64)
            for at in range(2 * len(self.link_cores[island])):
                there, first, _ = self.pair(island, at // 2)
                link = at - 2 * first  # its link toward `there`
                if link >= self.graph.between(island, there).count:
                    continue
                row = words[at]
                low = h["KM_CROSS_BYTES"]
                ring = h["KM_CROSS_ROW"] // 4  # the ring's words after the row
                rows[(link, island, there)] = {
                    **dict(
                        zip(
                            island_records.Crossings.FIELDS,
                            (int(row[f]) for f in fields),
                        )
                    ),
                    "bytes": int(row[low]) | int(row[low + 1]) << 32,
                    "ring_sending": int(row[ring + h["KM_PUT_RING_SENDING"] // 4]),
                    "ring_router": int(row[ring + h["KM_PUT_RING_ROUTER"] // 4]),
                    **{
                        f: int(row[ring + 4 + h[f"KM_PUT_{f.upper()}"] // 4])
                        for f in island_records.Crossings.IDLE
                    },
                    **(put_words(row, h) if puts else {}),
                }
        return island_records.Crossings(rows)

    @property
    def _forward(self):
        """KM_FORWARD over more than two islands: the link services forward (built only
        then, the kernels smaller without); KM_FAR_PUTS when a put crosses two links."""
        if len(self.graph.islands) <= 2:
            return {}
        diameter = self.graph.diameter()
        return {
            "KM_FORWARD": 1,
            # the rooms: a writer core and CPU, a hop level (km_abi.h KM_FORWARD_*)
            "KM_FORWARD_CORES": max(len(cores) for cores in self.link_cores),
            "KM_FORWARD_CPUS": max(min(link.count, 2) for link in self.graph.links),
            "KM_FORWARD_LEVELS": max(diameter - 1, 1),
            **({"KM_FAR_PUTS": 1} if diameter > 1 else {}),
        }

    def _hand_rectangles(self, spec):
        """Each class with a rectangle the islands load (`spec`, the classes it names
        ahead) gets island 0's NoCs and the islands' stack (no rect=: flat_rectangle),
        or on a split card the card's NoCs and each island's rows and ranks
        (island_rectangles), before the plan checks it."""
        if len(self.sessions) < 2:
            return
        if isinstance(spec, programs_module.Plan):
            listed = list(spec.programs)
        elif isinstance(spec, dict | list | tuple):
            listed = list(spec)
        else:
            listed = [spec]
        listed += [o for p in listed for o in getattr(p, "added_ahead_programs", ())]
        needing = [
            p
            for p in listed
            if isinstance(p, Program)
            and getattr(p, "rectangle", None)
            and p.rectangle["args"] is None
        ]
        if not needing:
            return
        if self.split > 1:
            nocs = self.sessions[0].nocs()
            rows = sizes._split_rows(self.graph, self.sessions[0].grid)
            split = tuple(
                (first, count, self.graph.count(island), self.graph.first(island))
                for island, (first, count) in enumerate(rows)
            )
            stack = self._stack([nocs] * len(rows), rows)
            for program in needing:
                if program.rectangle["nocs"] is None:
                    program.rectangle["nocs"] = nocs
                program.rectangle["split"] = split
                program.rectangle["stack"] = stack
            return
        nocs = self.sessions[0].nocs()
        stack = self._stack([nocs] + [s.nocs() for s in self.sessions[1:]], None)
        for program in needing:
            state = program.rectangle
            state["stack"] = stack
            if state["nocs"] is None:
                state["nocs"] = nocs

    def _stack(self, nocs, bands):
        """The islands one below the other in the island order, as a rectangle that
        spans them counts (dsl.program.stacked_rectangle): each island's NoCs, first
        flat rank and ranks; a split card's `bands` (first row, rows); each island's
        first row in the stack (a split card's: its first row of the card) and rows."""
        rows = [n.grid[1] for n in nocs] if bands is None else [r for _, r in bands]
        tops = [sum(rows[:i]) for i in range(len(rows))]
        if bands is not None:
            tops = [top for top, _ in bands]
        islands = [
            (one, self.graph.first(i), self.graph.count(i))
            for i, one in enumerate(nocs)
        ]
        return {"islands": islands, "bands": bands, "tops": tops, "rows": rows}

    def _hold_links(self, timeout=10.0):
        """Each link CPU of both islands opens its connection, held until its kernel
        ends, and says hello to the far link core's same CPU (an OPEN a link core,
        waited together): no send goes out before the far side runs."""
        tables = [list(run.backend.port.core_table()) for run in self.run.runs]
        if any(table != tables[0] for table in tables):
            raise KomorebiError(
                "the islands' core tables differ: a link core that pushes a rank's "
                "pages finds the far core in its own table"
            )
        waits = [
            (here, self._open_link(here, k))
            for here in range(len(self.graph.islands))
            for k in range(len(self.link_cores[here]))
        ]
        self.wait_together(waits, timeout)
        self.log(f"page links held, hello both ways: link cores {self.link_cores}")

    def _island_defines(self):
        """The defines a load gives every program the islands load (and the classes
        named ahead, built as added): a tensor's word may name its table entry (several
        islands); the link cores' CPUs serve PAGES (km_pages.hpp)."""
        out = {"KM_TENSOR_TABLE": 1} if len(self.sessions) > 1 else {}
        out |= self._table_bound() | self._islands_bound()
        if self.links:
            asks = {"KM_ASKS_BY_HOST": 1} if self.ASKS_BY == "host" else {}
            out |= {"KM_PAGE_LINKS": 1, **asks, **self._forward}
        return out

    def _island_loads(self, spec):
        """What each island's session loads for `spec`: a kernel-form Program on every
        island, a list of them one an island, or (the runtime's form) each island's
        part of the plan over flat ranks: its groups cut to its ranks, the same
        program fields."""
        count = len(self.sessions)
        if isinstance(spec, Program) and spec.form() != "program":
            return [spec] * count
        kernel_form = isinstance(spec, list | tuple) and all(
            isinstance(one, Program) and one.form() != "program" for one in spec
        )
        if kernel_form and spec:
            if len(spec) != count:
                raise ValueError(
                    f"a list of kernel-form Programs is one an island: {count} here, "
                    f"{len(spec)} given (several programs a run are in the runtime's "
                    "form, km_program)"
                )
            return list(spec)
        whole = programs_module.plan(spec, self.graph.ranks)  # checks it as one card
        parts = []
        for island in range(count):
            first, ranks = self.graph.first(island), self.graph.count(island)
            groups = []
            for group in whole.groups:
                low = max(group.first, first)
                high = min(group.first + group.count, first + ranks)
                if high > low:
                    groups.append(
                        programs_module.Group(low - first, high - low, group.programs)
                    )
            parts.append(
                programs_module.Plan(
                    whole.programs, tuple(groups), ranks, whole.owner, whole.switch
                )
            )
        return parts

    def _islands_bound(self):
        """KM_ISLANDS_MAX for more than ISLANDS_MAX islands: the per-island arrays'
        length on the cores. More than ROUND_ISLANDS_MAX islands are refused."""
        count = len(self.graph.islands) if self.graph is not None else 1
        if abi.headers()["KM_ISLANDS_MAX"] != ISLANDS_MAX:
            raise KomorebiError("ISLANDS_MAX differs from km_abi.h's")
        if count > ROUND_ISLANDS_MAX:
            raise KomorebiError(
                f"a load over {count} islands: a dispatcher's round of a run takes "
                f"the answers of at most {ROUND_ISLANDS_MAX} islands, a bit each; "
                f"use at most {ROUND_ISLANDS_MAX} islands"
            )
        return {"KM_ISLANDS_MAX": count} if count > ISLANDS_MAX else {}

    def _load_split(self, program, options):
        """load() on a card split into islands: one launch, the links wires (NoC
        writes to the far island's tiles), the services as on two cards."""
        if not isinstance(program, Program) or program.form() == "program":
            raise ValueError(
                "a card split into islands loads one Program of the kernel form: its "
                "one launch builds no runtime's-form programs (km_program, a list or "
                "a dict) yet"
            )
        self._hand_rectangles(program)
        asked, own = program, program.defines
        try:
            plans = None
            if self.links:
                (program,) = self._linked([program])
                asks = {"KM_ASKS_BY_HOST": 1} if self.ASKS_BY == "host" else {}
                # tensors held by island: a tensor's word may name its table entry
                table = {} if self.card_dram else {"KM_TENSOR_TABLE": 1}
                table |= {} if self.card_dram else self._table_bound()
                table |= self._islands_bound()
                program.defines = {
                    **program.defines,
                    "KM_PAGE_LINKS": 1,
                    "KM_WIRE": 1,
                    **self._forward,
                    **asks,
                    **table,
                }
                plans = self._links()
            tables = [0] if self.card_dram else self._tables_for_load()
            try:
                runs = self.sessions[0].load_islands(
                    program, self.graph, links=plans, tensors=tables[0], **options
                )
            except kernel_buffer.Resize as resize:  # the card opened again at it
                self._split_reopen(resize)
                plans = self._links() if self.links else None
                tables = [0] if self.card_dram else self._tables_for_load()
                runs = self.sessions[0].load_islands(
                    program, self.graph, links=plans, tensors=tables[0], **options
                )
            self.sessions[0]._kept.stop()  # a run loaded: no reopen after it
        except (KernelSizeError, L1BudgetError) as error:
            self._refused(error)  # kept, as over cards: the next run opens at it
            raise
        finally:
            program.defines = own
        if mock.active():  # every island built: a mock run for the launch
            self.run = mock.Run(
                parts=runs, host=self._mock_host(runs), graph=self.graph
            )
            self._remember(asked)
            return self.run
        self.run = island_run.IslandRun(
            self.graph,
            runs,
            linked=bool(self._regions),
            counts=self._crossings if self.links else None,
        )
        self.run.session = self  # its chains see card_dram (one tensor, no gather)
        self._remember(asked)
        if self.links:
            try:
                self._hold_links()
            except BaseException:
                self._aborted(self.run.runs)
                raise
        return self.run

    def _aborted(self, runs):
        """Every island's run of a load that failed aborted, each even if another's
        abort raised; such an error is logged, the load's own is the one raised."""
        failure = island_run.abort_every(runs)
        if failure is not None:
            self.log(
                f"a failed load's abort raised {type(failure).__name__}: {failure}"
            )

    def _follow(self, error):
        """Raises kernel_buffer.Resize when two cards follow their kernels (no
        kernel_bytes given) and an island's are over the buffer, and the cards can be
        opened again under the session's tensors (_not_reopened); else says why not."""
        total = getattr(getattr(error, "sizes", None), "total", 0)
        if not (
            self._kernel_followed
            and len(self.cards) > 1
            and isinstance(error, KernelSizeError)
            and total
        ):
            return
        size = kernel_buffer.need(total)
        why = self._not_reopened(size)
        if why is not None:
            self.log(f"kernel buffer: the cards are not opened again in the run: {why}")
            return
        raise kernel_buffer.Resize(size, f"the kernels take {total:,} B")

    def _not_reopened(self, size):
        """Why the cards cannot be opened again at a kernel buffer of `size` bytes, or
        None: over the most, an island that loaded a run (its tensors may be written),
        or an L1 buffer of an island under the larger buffer's end (checked first)."""
        if size > sizes.KERNEL_BUFFER_MAX:
            return f"{size:,} B is over the most, {sizes.KERNEL_BUFFER_MAX:,} B"
        module = native.load_module(self.tree)
        for island, session in enumerate(self.sessions):
            refused = session._kept.refused()
            if refused is not None:
                return f"island {island}: {refused}"
            kernel, free, top = (int(v) for v in module.kernel_buffer(session.device))
            end = top - free + size - kernel  # the L1 buffers' base at `size`
            lowest = int(module.lowest_l1(session.device))
            if lowest and lowest < end:
                return (
                    f"island {island}: an L1 buffer at {lowest:#x} is under the end "
                    f"of a {size:,} B kernel buffer ({end:#x}); free it before the "
                    "load, or give the session kernel_bytes"
                )
        return None

    def _cards_reopen(self, resize):
        """The cards opened again at the kernel buffer `resize` names, as one card's
        session is (kernel_buffer.reopen): each island's device and the mesh closed,
        the mesh and the X280 runtime opened at it, each island's session given its
        new device, its kept tensors made again at their addresses and written."""
        started = time.perf_counter()
        before, size = self.kernel_bytes, resize.size
        self.log(
            f"kernel buffer {before:,} -> {size:,} B ({resize.why}): the cards are "
            "closed and opened again"
        )
        for session in self.sessions:
            # its channel's mapping goes with the device: mapped again at the first
            # transfer (the rewrite), as a session given its device maps it
            session.channel, session._channel_mapped = None, False
            session.device.close()
            session.device = None
        self.mesh.close()
        self.kernel_bytes, self._worker_l1 = size, sizes.worker_l1_size(size)
        module = native.load_module(self.tree)
        self.mesh = module.Islands(len(self.cards), self._worker_l1, self.packet)
        if not mock.active():  # a mock has no X280
            self.mesh.open_x280()
        made = written = 0
        for island, session in enumerate(self.sessions):
            session.device = self.mesh.device(island)
            session._kept.remake(session)
            tensors = session._kept.live()
            made += len(tensors)
            written += sum(kernel_buffer.rewrite(session, tensor) for tensor in tensors)
        if self.links:
            self._open_links(module)
        self.log(
            f"kernel buffer {size:,} B: cards opened again, {made} tensor(s) made "
            f"again and {written:,} B written ({time.perf_counter() - started:.2f} s)"
        )

    def _split_reopen(self, resize):
        """A split card's load whose kernels need another kernel buffer: the card
        opened again at it, its tensors made again (kernel_buffer.reopen), as a card's
        own session does; the load is then made again."""
        session = self.sessions[0]
        kernel_buffer.reopen(
            session, resize.size, sizes.worker_l1_size(resize.size), resize.why
        )
        self.kernel_bytes = session.kernel_bytes = resize.size
        self._worker_l1 = session._worker_l1

    def _mock_host(self, runs):
        """The IslandRun over the islands' mock runs' own Runs (their chains built as
        the card's); None if an island has none."""
        hosts = [getattr(run, "host", None) for run in runs]
        if any(host is None for host in hosts):
            return None
        host = island_run.IslandRun(self.graph, hosts, linked=bool(self._regions))
        host.session = self
        return host

    def _refused(self, error):
        """A load refused, its kernels over this session's buffer or its L1 table
        failed below a larger buffer than they need: their size kept, so that the
        script's next run opens at it; the log says so."""
        total = getattr(getattr(error, "sizes", None), "total", 0) or max(
            (one._last_total for one in self.sessions), default=0
        )
        if self._memory is None or not total:
            return
        need = kernel_buffer.need(total)
        smaller = isinstance(error, L1BudgetError) and need < self.kernel_bytes
        if not (smaller or isinstance(error, KernelSizeError)):
            return
        self._memory.wanted(total, smaller)
        self.log(
            f"kernel buffer: the kernels take {total:,} B, this session's buffer is "
            f"{self.kernel_bytes:,} B; remembered, the script's next run opens at "
            f"{need:,} B"
        )

    def _remember(self, program):
        """The kernel buffer the islands' load needed (its largest), kept for the
        script's next run."""
        total = max((one._last_total for one in self.sessions), default=0)
        if self._memory is not None and total:
            self._memory.loaded(program, total)

    def _remembered(self, expect):
        """The kernel buffer this session opens at: the size remembered for the script
        (or `expect`, a made Program) with headroom, else KERNEL_BYTES; a message
        service a link keeps KERNEL_BYTES_SERVICES (message_services)."""
        shape = f"islands {len(self.cards)} cards x {self.split}"
        self._memory = remembered.Memory(
            self.log, expect, shape, default=f"{KERNEL_BYTES:,} B", room=True
        )
        size = self._memory.opened or KERNEL_BYTES
        if self.message_links:
            size = max(size, KERNEL_BYTES_SERVICES)
        return min(size, sizes.KERNEL_BUFFER_MAX)

    def load(self, program, *args, **options):
        """Loads `program` on every island in turn, with Session.load's options (by
        position too); an IslandRun over flat ranks. As on one card: a Program, or the
        runtime's form (a list, or {program: (first, count)} in flat ranks); or a list
        of kernel-form Programs, one an island. Each brings its own X280 threads."""
        if len(args) > len(LOAD_OPTIONS):
            raise TypeError(
                f"load: {len(args)} options by position, at most "
                f"{len(LOAD_OPTIONS)}"
            )
        for name, value in zip(LOAD_OPTIONS, args):
            if name in options:
                raise TypeError(f"load: option {name!r} given twice")
            options[name] = value
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if self.split > 1:  # one launch on the card, a dispatcher an island
            return self._load_split(program, options)
        try:
            return self._load_cards(program, options)
        except kernel_buffer.Resize as resize:  # the cards opened again at it
            self._cards_reopen(resize)
        return self._load_cards(program, options)

    def _load_cards(self, program, options):
        """load() over cards (or one card's one island): each island's Session.load
        in turn, an IslandRun over them; kernel_buffer.Resize when the cards follow
        kernels over their buffer (_follow)."""
        self._hand_rectangles(program)
        programs = self._island_loads(program)
        if self.ASKS_BY not in ("core", "host"):
            raise ValueError(f"ASKS_BY={self.ASKS_BY!r}: 'core' or 'host'")
        if self.links and all(isinstance(one, Program) for one in programs):
            programs = self._linked(programs)
        owned = {}  # every Program the islands load, once: the island defines on each
        for one in programs:
            for each in (
                one.programs if isinstance(one, programs_module.Plan) else [one]
            ):
                owned[id(each)] = each
                for other in getattr(each, "added_ahead_programs", ()):
                    owned[id(other)] = other  # its parts built at the load, as added
        owned = list(owned.values())
        defines = [one.defines for one in owned]
        # an X280 thread's copies name tensor words too (kmx.h): the same bound
        threads = [t for one in owned for t in getattr(one, "threads", {}).values()]
        thread_defines = [t.defines for t in threads]
        runs = []
        try:
            for one, own in zip(owned, defines):
                one.defines = {**own, **self._island_defines()}
            for thread, own in zip(threads, thread_defines):
                thread.defines = {**own, **self._table_bound()}
            beside = bool(threads) and options.get("thread", "card") == "card"
            for session in self.sessions:  # one upload region and link area on each
                session._rings_beside = beside and len(programs) > 1
            plans = self._links(programs)
            tables = self._tables_for_load()
            for island, (session, one) in enumerate(zip(self.sessions, programs)):
                try:
                    # one island: the card's run as it is without islands
                    island_of = (self.graph, island) if len(programs) > 1 else None
                    runs.append(
                        session.load(
                            one,
                            island=island_of,
                            link=plans[island],
                            tensors=tables[island],
                            **options,
                        )
                    )
                except KomorebiError as error:
                    raise island_run._named(error, island) from None
            check_link_queues(
                [getattr(run.backend, "link_queue_at", None) for run in runs]
            )
            if mock.active():  # every island built: a mock run for the launch
                self.run = mock.Run(
                    parts=runs, host=self._mock_host(runs), graph=self.graph
                )
                self._remember(program)
                return self.run
            self._check_links(runs, plans)
        except (KernelSizeError, L1BudgetError) as error:
            self._aborted(runs)
            self._follow(error)  # the cards opened again at the kernels' size
            self._refused(error)
            raise
        except BaseException:
            self._aborted(runs)
            raise
        finally:
            for one, own in zip(owned, defines):
                one.defines = own
            for thread, own in zip(threads, thread_defines):
                thread.defines = own
            for session in self.sessions:
                session._rings_beside = False
        self.run = island_run.IslandRun(
            self.graph,
            runs,
            linked=bool(self._regions),
            counts=self._crossings if self.links else None,
        )
        self.run.session = self
        self._remember(program)
        if self.links:
            try:
                self._hold_links()
            except BaseException:
                self._aborted(self.run.runs)
                raise
        return self.run
