"""The island session's load: one program over its islands (a split card, cards)."""

from collections.abc import Callable
from typing import Any

import numpy as np

from komorebi import abi
from komorebi.device import kernel_buffer, mock, remembered
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
)
from komorebi.program import several as programs_module
from komorebi.program.description import Program
from komorebi.session import sizes

# Two cards' kernel buffer: the largest build, a dynamic program with a message service
# a link (about 159,400 B), in steps of 16 KiB. kernel_bytes= overrides it.
KERNEL_BYTES = 160 * 1024


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

    def _crossings(self, timeout=10.0):
        """Crossings read now: a COUNTS command to every link core of each island (each
        CPU writes its row into the island's counts tensor), waited for together."""
        h = abi.headers()
        waits = []
        for island in range(len(self.graph.islands)):
            words = [0] * 10
            words[h["KM_PAGES_TO"]] = self._counts[island].address
            words[h["KM_PAGES_PAGE"]] = CROSS_PAGE
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
            (raw,) = self.sessions[island].read(self._counts[island])
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
        ahead) gets island 0's NoCs and the islands' rank count (rank_rectangle; counts
        that differ refuse it by name), or on a split card the card's NoCs and each
        island's rows and ranks (island_rectangles), before the plan checks it."""
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
            for program in needing:
                if program.rectangle["nocs"] is None:
                    program.rectangle["nocs"] = nocs
                program.rectangle["split"] = split
            return
        counts = {len(cores) for cores in self.graph.rank_cores}
        ranks = next(iter(counts)) if len(counts) == 1 else None
        nocs = self.sessions[0].nocs()
        for program in needing:
            state = program.rectangle
            if state["rect"] is None and ranks is None:
                raise KomorebiError(
                    f"{program.name}: its default rectangle is an island's rank cores, "
                    f"and the islands have {sorted(counts)}: make it with rect="
                )
            if state["nocs"] is None:
                state["nocs"] = nocs
            state["ranks"] = ranks

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
        length on the cores."""
        count = len(self.graph.islands) if self.graph is not None else 1
        if abi.headers()["KM_ISLANDS_MAX"] != ISLANDS_MAX:
            raise KomorebiError("ISLANDS_MAX differs from km_abi.h's")
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
            runs = self.sessions[0].load_islands(
                program, self.graph, links=plans, tensors=tables[0], **options
            )
        except (KernelSizeError, L1BudgetError) as error:
            self._refused(error)  # kept, as over cards: the next run opens at it
            raise
        finally:
            program.defines = own
        if mock.active():  # every island built: a mock run for the launch
            self.run = mock.Run(parts=runs)
            self._remember(asked)
            return self.run
        self.run = island_run.IslandRun(
            self.graph,
            runs,
            linked=bool(self._regions),
            counts=self._crossings if self.links else None,
        )
        self._remember(asked)
        if self.links:
            try:
                self._hold_links()
            except BaseException:
                self.run.abort()
                raise
        return self.run

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

    def load(self, program, **options):
        """Loads `program` on every island in turn, with Session.load's options; an
        IslandRun over flat ranks. As on one card: a Program, or programs in the
        runtime's form (a list, or {program: (first, count)} in flat ranks); or a list
        of kernel-form Programs, one an island. Each brings its own X280 threads."""
        if self.run is not None and not self.run.stopped:
            raise RuntimeError("a run is loaded: stop it first")
        if self.split > 1:  # one launch on the card, a dispatcher an island
            return self._load_split(program, options)
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
                self.run = mock.Run(parts=runs)
                self._remember(program)
                return self.run
            self._check_links(runs, plans)
        except (KernelSizeError, L1BudgetError) as error:
            for run in runs:
                run.abort()
            self._refused(error)
            raise
        except BaseException:
            for run in runs:
                run.abort()
            raise
        finally:
            for one, own in zip(owned, defines):
                one.defines = own
            for thread, own in zip(threads, thread_defines):
                thread.defines = own
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
                self.run.abort()
                raise
        return self.run
