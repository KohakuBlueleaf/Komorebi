"""The island session's calls between islands: sends, asks, crossings, waits."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from komorebi import abi
from komorebi.device import chips, dram
from komorebi.errors import (
    KomorebiError,
    WaitTimeout,
)
from komorebi.islands import records as island_records
from komorebi.islands import run as island_run

SLICE_S = 0.0  # an island's turn in wait_together: one native round (IslandRun)


class CallsMixin:
    """The island session's calls between islands: sends, asks, crossings, waits."""

    # the state of the session's other parts this one reads or sets, owned there
    ASKS_BY: str  # IslandSession
    log: Callable  # IslandSession
    _regions: list  # MeshMixin
    channels: tuple  # MeshMixin
    seconds: tuple  # MeshMixin
    graph: Any  # MeshMixin
    arch: str  # MeshMixin
    link_cores: tuple  # MeshMixin
    links: int  # MeshMixin
    packet: int  # IslandSession
    run: Any  # MeshMixin
    sessions: tuple  # MeshMixin

    def _page_fits(self, page):
        """Refuses pages of `page` bytes larger than the links' packet (a page crosses
        in one packet: a link CPU gives up on a larger one)."""
        if page > self.packet:
            raise ValueError(
                f"pages of {page:,} B do not cross: a page crosses in one packet, and "
                f"this session's packet is {self.packet:,} B (IslandSession packet=)"
            )

    def _far_dram(self, what):
        """Refuses `what`, pages sent into a far island's DRAM, on a chip whose link
        packets may not target the far chip's DRAM (chips link_far_dram)."""
        if not chips.chip(self.arch).link_far_dram:
            raise ValueError(
                f"{what}: pages into another island's DRAM do not cross on {self.arch} "
                "(docs/reference/hazards.md#a-link-packet-for-a-far-wormhole-chips-"
                "dram); send them into a far core's L1 (send_pages(core=...))"
            )

    def _across(self, here, there):
        """The link core of `there` that counts the pages `here` sends it."""
        return self.pair(here, self._toward(here, there)[0])[2]

    def _open_link(self, here, k):
        """Queues the OPEN of link core k of island `here` (its CPUs' connections,
        then a hello to the same CPUs of the far link core across, self.pair).
        Returns its handle."""
        h = abi.headers()
        there, first, far = self.pair(here, k)
        buffer, chunk = self.staging(here)
        words = [0] * 10
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_TO_CORE"]] = chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, h["KM_PAGES_HELLO_PORT"])
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, far) | 1 << 16 | h["KM_PAGES_OPEN"] << 24
        )
        words[h["KM_PAGES_LINKS"]] = self.links_word(here, there)
        if self.seconds:  # a thin second sender's: each link core's second link
            words[h["KM_PAGES_SIDES"]] = self.seconds_word(here, there)
        words[h["KM_PAGES_BUFFER"]] = buffer
        return self.run.island(here)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(here, k, 1), words
        )

    def _send(
        self,
        here,
        there,
        start,
        page,
        first,
        pages,
        to,
        core=None,
        counted=None,
        port=None,
        shape=0,
    ):
        """Queues the send by `here`'s link cores toward `there` of pages first .. of
        the DRAM buffer at `start` (aligned `page` bytes): into the same pages of the
        buffer at `to`, or (`core`) L1 `to` of that core on; send_pages' counts."""
        h = abi.headers()
        self._page_fits(page)
        k, cores = self._toward(here, there)
        counted = counted if counted is not None else core
        counted = counted if counted is not None else self.pair(here, k)[2]
        port = h["KM_PORTS"] - 1 if port is None else port
        buffer, chunk = self.staging(here)
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = start
        words[h["KM_PAGES_PAGE"]] = page
        words[h["KM_PAGES_FIRST"]] = first
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to
        target = h["KM_PAGES_DRAM"] if core is None else self.noc_of(there, core)
        words[h["KM_PAGES_TO_CORE"]] = target | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(there, port)
        words[h["KM_PAGES_COUNTER"]] = (
            self.noc_of(there, counted) | 1 << 16 | h["KM_PAGES_SEND"] << 24
        )
        words[h["KM_PAGES_LINKS"]] = self.links_word(here, there)
        if not self._regions:  # no message service holds link 0: it carries pages too
            shape |= h["KM_PAGES_SHAPE_LINK0"]
        words[h["KM_PAGES_BUFFER"]] = buffer | shape << h["KM_PAGES_SHAPE_SHIFT"]
        handle = self.run.island(here)._queue_command(
            h["KM_HANDLER_PAGES"], self.graph.link_group(here, k, cores), words
        )
        return here, handle

    def _toward(self, here, there):
        """(first, count) of `here`'s link cores whose links go to island `there`."""
        if there not in self.graph.neighbours(here):
            raise ValueError(f"islands {here} and {there} are not neighbours")
        if len(self.graph.neighbours(here)) == 1:
            return 0, len(self.link_cores[here])
        return self.graph.link_cores_toward(here, there)

    def ask(self, tensor, first, pages, island, core, to, port):
        """Pages first .. of `tensor`, held by another island, into L1 address `to` of
        local core `core` of `island`, counted at its port `port`: what that core waits
        on (km::port_wait). Returns (holder, handle) of the holder's send."""
        if self.ASKS_BY != "host":
            raise KomorebiError(
                "session.ask(): this session's calls ask for their own pages "
                f"(IslandSession.ASKS_BY = {self.ASKS_BY!r}); the host asks only for "
                "a session whose ASKS_BY is 'host', set before its load"
            )
        holder = tensor.holders(first, pages)[0][0]
        if holder == island:
            raise ValueError(f"pages {first} + {pages}: island {island} holds them")
        return self.send_pages(tensor, first, pages, to, core, core, port, island)

    def ask_range(self, tensor, first, pages, group, to, port):
        """One ask for every rank of `group` (first rank, ranks; one island): rank first
        + i gets pages first + i * pages .. of `tensor` (held by another island) into L1
        address `to`, counted at port `port`. One command to the holder's link cores,
        which split the ranks among their CPUs. Returns (holder, handle)."""
        h = abi.headers()
        if self.ASKS_BY != "host":
            raise KomorebiError(
                "session.ask_range(): this session's calls ask for their own pages "
                f"(IslandSession.ASKS_BY = {self.ASKS_BY!r}); the host asks only for "
                "a session whose ASKS_BY is 'host', set before its load"
            )
        rank, ranks = group
        island, local = self.graph.resolve(rank)
        if ranks < 1 or self.graph.resolve(rank + ranks - 1)[0] != island:
            raise ValueError(f"group {group}: not one island's ranks")
        holders = tensor.holders(first, pages * ranks)
        if len(holders) != 1 or holders[0][0] == island:
            raise ValueError(
                f"pages {first} + {pages * ranks}: held by islands {holders}, not by "
                f"one other than island {island}"
            )
        holder, _, _, index = holders[0]
        self._page_fits(tensor.aligned_page_bytes)
        if not 0 <= port < h["KM_PORT_SLOTS"] or ranks > 0xFFFF:
            raise ValueError(f"port {port}, ranks {ranks}")
        buffer, chunk = self.staging(holder)
        words = [0] * 10
        words[h["KM_PAGES_FROM"]] = tensor.addresses[holder]
        words[h["KM_PAGES_PAGE"]] = tensor.aligned_page_bytes
        words[h["KM_PAGES_FIRST"]] = index
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_TO"]] = to
        core = self.graph.rank_cores[island][local]
        words[h["KM_PAGES_TO_CORE"]] = core | chunk // 64 << 16
        words[h["KM_PAGES_COUNT"]] = self.port_address(island, port)
        words[h["KM_PAGES_COUNTER"]] = ranks | 1 << 16 | h["KM_PAGES_RANGE"] << 24
        words[h["KM_PAGES_LINKS"]] = self.links_word(holder, island)
        words[h["KM_PAGES_BUFFER"]] = buffer
        handle = self.run.island(holder)._queue_command(
            h["KM_HANDLER_PAGES"],
            self.graph.link_group(holder, *self._toward(holder, island)),
            words,
        )
        return holder, handle

    def core_group(self, island, core):
        """The group of local core `core` alone in a command to `island`: a link core
        by its name (Graph.link_group), a rank's core by its flat rank."""
        if core in self.link_cores[island]:
            k = self.link_cores[island].index(core)
            return self.graph.link_group(island, k, 1)
        if core not in self.graph.rank_cores[island]:
            raise ValueError(f"island {island} has no core {core}")
        return self.graph.first(island) + self.graph.rank_cores[island].index(core), 1

    def cross(self, source, target, first=0, pages=None, timeout=10.0, shape=0):
        """Writes pages first .. of `source` into the same pages of `target` where
        another island holds them (a copy target: every other island's copy), sent
        over the links, waited for on the far island; `shape` as send_pages. Returns
        pages, bytes and seconds, the submit to the last count."""
        self._far_dram("session.cross()")
        start, count = dram.page_range(first, pages, source.pages)
        sends = []
        for here, low, length, _ in source.holders(start, count):
            for there in range(len(self.sessions)):
                if there != here:
                    sends += [
                        (holder, part, part_length)
                        for holder, part, part_length, _ in target.placement.runs(
                            low, length, there
                        )
                        if holder == there
                    ]
        started = time.perf_counter()
        waits, far = [], []
        for there, low, length in sends:
            here = source.holders(low, length)[0][0]
            counted = self.pair(here, self._toward(here, there)[0])[2]
            far.append((there, self.wait_pages(there, counted, length)))
            waits.append(
                self.send_pages(source, low, length, target, there=there, shape=shape)
            )
        # the senders first: a failure names the side that failed
        self.wait_together(waits + far, timeout)
        seconds = time.perf_counter() - started
        moved = sum(length for _, _, length in sends)
        return {
            "pages": moved,
            "bytes": moved * source.aligned_page_bytes,
            "seconds": seconds,
        }

    def links_word(self, here, there):
        """The PAGES word of `here`'s links toward neighbour `there`: their channels, a
        byte each from its first link core's CPU 0, KM_PAGES_NO_LINK past them."""
        none = abi.headers()["KM_PAGES_NO_LINK"]
        first, cores = self.graph.link_cores_toward(here, there)
        per = self.graph.per
        mine = list(self.channels[here][per * first : per * (first + cores)])
        return int.from_bytes(bytes((mine + [none] * self.LINKS_MAX)[:4]), "little")

    def seconds_word(self, here, there):
        """The OPEN word of `here`'s second links toward `there`: a byte each link
        core from its first's (KM_PAGES_SIDES), KM_PAGES_NO_LINK where it has none."""
        none = abi.headers()["KM_PAGES_NO_LINK"]
        first, cores = self.graph.link_cores_toward(here, there)
        mine = list(self.seconds[here][first : first + cores])
        return int.from_bytes(bytes((mine + [none] * self.LINKS_MAX)[:4]), "little")

    def pair(self, island, k):
        """Link core `k` of `island`: (the island its links go to, the first link core
        toward it, the far island's link core across from this one). A neighbour's
        links take ceil(links / per) link cores in neighbour order, a link a CPU."""
        for there in self.graph.neighbours(island):
            first, cores = self.graph.link_cores_toward(island, there)
            if first <= k < first + cores:
                far = self.graph.link_cores_toward(there, island)[0] + k - first
                return there, first, self.link_cores[there][far]
        raise ValueError(f"island {island}: no link core {k}")

    def port_address(self, island, port):
        """The L1 address of port `port`'s count, the same on every core of `island`
        (KM_PORT_PAGES and the one after it: the page calls' own)."""
        h = abi.headers()
        if not 0 <= port < h["KM_PORT_SLOTS"]:
            raise ValueError(f"port {port} of {h['KM_PORT_SLOTS']}")
        control = self.run.island(island).backend.control_addr
        return control + h["KM_CONTROL_PORT"] + port * h["KM_PORT_STRIDE"]

    def reset_link(self, island, k, timeout=10.0):
        """Resets link core k of `island` by the host and opens its connections again:
        each closed at its router first, the core reloaded, then its OPEN with the far
        link core's hello taken as said (the far side runs). {island, core, by, seconds,
        reset_seconds, open_seconds}; KomorebiError if a step did not finish."""
        h = abi.headers()
        if (
            self.run is None
            or not self.links
            or not 0 <= k < len(self.link_cores[island])
        ):
            raise ValueError(f"island {island}: no link core {k} in a run with links")
        run = self.run.island(island)
        core = self.link_cores[island][k]
        none = h["KM_PAGES_NO_LINK"]
        per = self.graph.per
        held = [
            per * k + cpu
            for cpu in range(per)
            if per * k + cpu < len(self.channels[island])
            and self.channels[island][per * k + cpu] != none
        ]
        # a thin second sender's second link: the writer's (its hello port the next)
        second = [self.seconds[island][k]] if self.seconds else []
        second = [channel for channel in second if channel != none]
        drops = []
        areas = getattr(self, "_areas", None)
        if k == 0 and areas:  # lines given up with their commands: never sent
            area = areas[island]
            for name in ("ASK", "PUT"):
                drops += [
                    area + h[f"KM_{name}_TAIL"],
                    area + h[f"KM_{name}_TAKEN"],
                    area + h[f"KM_{name}_LINES"],
                    h["KM_ASK_SLOTS"],
                ]
        started = time.perf_counter()
        try:
            reset = run.reset_core(
                core,
                timeout,
                channels=[self.channels[island][link] for link in held] + second,
                drops=drops,
            )
            group = self.graph.link_group(island, k, 1)
            run.wait(run._queue_command(h["KM_HANDLER_NOP"], group), timeout)
            opened = time.perf_counter()
            backend = run.backend
            module, columns = backend.card.module, backend.port.grid_x
            cpus = [link - per * k for link in held] + [1] * len(second)
            for cpu in cpus:  # the far CPU said hello at the load: taken as heard
                port = self.port_address(island, h["KM_PAGES_HELLO_PORT"] + cpu)
                module.write_l1(
                    backend.device, core % columns, core // columns, port, [2]
                )
            run.wait(self._open_link(island, k), timeout)
        except KomorebiError as error:
            raise island_run._named(error, island) from None
        done = time.perf_counter()
        self.log(
            f"island {island}: link core {k} (core {core}) reset and opened again: "
            f"reset {(opened - started) * 1e3:.1f} ms, "
            f"open {(done - opened) * 1e3:.3f} ms"
        )
        return {
            "island": island,
            "core": core,
            "by": "host",
            "seconds": done - started,
            "reset_seconds": opened - started,
            "open_seconds": done - opened,
            "parts": reset["parts"],
        }

    def send_pages(
        self,
        source,
        first,
        pages,
        to,
        core=None,
        counted=None,
        port=None,
        there=None,
        shape=0,
    ):
        """Sends pages first .. of `source` by its holder's link cores into island
        `there` (default: the neighbour): the same pages of tensor `to`, or (`core`) L1
        from address `to`; each batch adds to port `port` of `counted` (default `core`,
        else the far link core). `shape`: the loop shape. Returns (island, handle)."""
        if not self.links:
            raise KomorebiError(
                "no page links: pages cross in a session over two cards"
            )
        if core is None:
            self._far_dram("session.send_pages() into a tensor")
        holders = source.holders(first, pages)
        if len(holders) != 1:
            raise ValueError(f"pages {first} + {pages}: held by islands {holders}")
        here, _, _, index = holders[0]
        if there is None:
            if len(self.graph.neighbours(here)) != 1:
                raise ValueError(f"island {here} has more neighbours: name `there`")
            there = self.graph.neighbours(here)[0]
        k, cores = self._toward(here, there)
        if core is None and (
            not isinstance(to, island_records.IslandTensor)
            or to.aligned_page_bytes != source.aligned_page_bytes
        ):
            raise ValueError("a tensor target has the source's aligned page bytes")
        if core is None and to.placement.runs(first, pages, there) != [
            (there, first, pages, index)
        ]:
            raise ValueError(
                f"pages {first} + {pages}: island {there} does not hold them at the "
                f"indices island {here} does ({index} on)"
            )
        return self._send(
            here,
            there,
            source.addresses[here],
            source.aligned_page_bytes,
            index,
            pages,
            to.addresses[there] if core is None else to,
            core,
            counted,
            port,
            shape,
        )

    def staging(self, island):
        """The L1 every core of `island` moves transfers through: (address, bytes of a
        chunk); two chunks. Pages asked from another island land there."""
        transfer = self.run.island(island).backend.transfer
        if transfer is None:
            raise KomorebiError(f"island {island}: the run has no transfer buffer")
        return transfer.address, transfer.chunk

    def wait_pages(self, island, core, pages, port=None):
        """Queues on local core `core` of `island` a wait (bounded) until port `port`'s
        count has reached `pages`, which it then takes off. Returns the handle."""
        h = abi.headers()
        port = h["KM_PORTS"] - 1 if port is None else port
        words = [0] * 10
        words[h["KM_PAGES_PAGES"]] = pages
        words[h["KM_PAGES_COUNT"]] = self.port_address(island, port)
        words[h["KM_PAGES_COUNTER"]] = h["KM_PAGES_WAIT"] << 24
        return self.run.island(island)._queue_command(
            h["KM_HANDLER_PAGES"], self.core_group(island, core), words
        )

    def wait_together(self, waits, timeout=10.0):
        """Waits, within one `timeout`, for [(island, handle)] of the islands' own runs:
        every island's queued commands stored first (run.send), then one native round
        a waited island in turn, as a wait of the run does (pages crossing need both
        islands'). Returns the events in the order given; errors name their island."""
        mine = {}
        for island, handle in waits:
            mine.setdefault(island, []).append(handle)
        pending = {i: np.unique(np.asarray(h, np.int64)) for i, h in mine.items()}
        deadline = time.monotonic() + timeout
        # every island stored, not only the waited ones: a command queued before may
        # cross to islands not waited here, and it needs every part started
        self.run.send()
        while any(handles.size for handles in pending.values()):
            if time.monotonic() > deadline:
                island = next(i for i, h in pending.items() if h.size)
                raise island_run._named(
                    WaitTimeout(
                        f"{pending[island].size} handles not done in {timeout} s",
                        pending[island].tolist(),
                    ),
                    island,
                )
            for island, handles in pending.items():
                if handles.size:
                    run = self.run.island(island)
                    try:
                        run._check_alive()
                        run._step(handles, int(handles[-1]), SLICE_S)
                    except KomorebiError as error:
                        raise island_run._named(error, island) from None
                    pending[island] = handles[~run._arrived(handles)]
        events = {}
        for island, handles in mine.items():
            try:
                got = self.run.island(island).wait(handles, island_run._left(deadline))
            except KomorebiError as error:
                raise island_run._named(error, island) from None
            events.update({(island, h): e for h, e in zip(handles, got)})
        return [events[(island, handle)] for island, handle in waits]
