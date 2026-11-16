"""X280 threads, ports and ranks of a load: the ports' lines, the ranks' words, the
regions of the work clusters and the threads' rings."""

from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi.device import dram
from komorebi.device.dram import DRAM_BANKS
from komorebi.errors import StartError
from komorebi.program import several as programs
from komorebi.program import thread as thread_module
from komorebi.program.description import messages_header

X280_QUEUES = "x280_queues"  # the region of the X280 ranks' queues in their thread
REGION_WHOLE = 4 << 20  # a region up to this size is read in one read, else in chunks
REGION_CHUNK = 2 << 20  # of this size, with a log line each


def _peers(session, dispatch, images):
    """{cluster: komorebi.program.thread.Peers} for the threads on the card: each with
    the clusters running its image, their tiles and their banks' DRAM tiles."""
    out, nonce = {}, thread_module.new_check_value() & 0xFFFFFFFF
    for cluster, group in thread_module.peer_groups(images).items():
        tiles, banks = {}, {}
        for peer in group:
            info = dispatch.work_info(peer)
            tiles[peer] = info["virtual_x"] | info["virtual_y"] << 16
            x, y = session.device.dram_core(thread_module.BANK_OF_CLUSTER[peer])
            banks[peer] = int(x) | int(y) << 16
        out[cluster] = thread_module.Peers(nonce, min(group), tiles, banks)
    return out


class _PortLines(NamedTuple):
    """Each core's tile by core index, and {core: (X280 tile, mailbox address)}."""

    cores: list
    lines: dict


def _port_lines(program, dispatch, rings, cores=None):
    """The program's X280 ports over the first `cores` cores (default the grid):
    ({core: (cluster, cpu)}, _PortLines), or (None, None) when it declares none."""
    if program.x280_ports is None:
        return None, None
    cores = cores or dispatch.grid_x * dispatch.grid_y
    plan = thread_module.port_plan(program.x280_ports["serve"], cores, set(rings))
    lines = {}
    for core, (cluster, _) in plan.items():
        info = dispatch.work_info(cluster)
        tile = info["virtual_x"] | info["virtual_y"] << 16
        lines[core] = (
            tile,
            thread_module.port_box(info["lim_base"], rings[cluster][0], core),
        )
    return plan, _PortLines(
        [int(core_tile) for core_tile in dispatch.core_table()[:cores]], lines
    )


def _x280_ranks(program, owner, placed, on_card):
    """The thread of the program's X280 ranks, its queues' region (X280_QUEUES) added;
    None when the program has none."""
    messages = program.messages
    if messages is None or messages.x280_ranks == 0:
        return None
    work = owner.threads.get(messages.x280_cluster)
    if placed is not None or work is None or not on_card:
        raise ValueError(
            f"{program.name}: X280 ranks need one program with its thread on cluster "
            f"{messages.x280_cluster}, on the card"
        )
    size = messages.queue_region_bytes()
    if work.regions.get(X280_QUEUES, size) != size:
        raise ValueError(f"{program.name}: the thread's region {X280_QUEUES!r} is ours")
    work.regions[X280_QUEUES] = size
    return work


def _x280_words(session, program, dispatch, rings, regions, plan):
    """The X280 ranks' words of km::MessagesArgs (without the inbox): their queues'
    bank address and DRAM tile, their cluster's tile, the cores' credit lines there."""
    messages = program.messages
    cluster = messages.x280_cluster
    bank, queue = regions[thread_module.BANK_OF_CLUSTER[cluster]].address(
        (cluster, X280_QUEUES)
    )
    x, y = session.device.dram_core(bank)
    info = dispatch.work_info(cluster)
    at = thread_module.ranks_at(rings[cluster][0], plan)
    return {
        "x280_queue": queue,
        "x280_bank": int(x) | int(y) << 16,
        "x280_tile": info["virtual_x"] | info["virtual_y"] << 16,
        "x280_credits": (
            info["lim_base"] + at + thread_module.layout()["KMX_RANKS_CREDITS"]
        ),
    }


def _x280_line(program, dispatch, backend, words, wait, mhz):
    """The X280 ranks' line and area for their thread
    (komorebi.program.thread.Ranks)."""
    messages = program.messages
    abi_headers, (constants, _) = abi.headers(), messages_header()
    cores = dispatch.grid_x * dispatch.grid_y
    ticks = thread_module.TIMER_HZ
    line = {
        "ranks": messages.x280_ranks,
        "cores": cores,
        "queue": words["x280_queue"],
        "cap": messages.queue_messages,
        "message_bytes": messages.message_bytes,
        "quota": messages.quota(cores),
        "counters": messages.counters,
        "idle": max(1, round(messages.idle_cycles / (mhz * 1e6) * ticks)),
        "inbox": backend.inbox.address,
        "inbox_messages": constants["kMessagesInbox"],
        "lines": dispatch.lim_base + abi_headers["KMD_LIM_MESSAGES"],
        "lines_tile": dispatch.l2_x | dispatch.l2_y << 16,
        "wait": round(wait * ticks),
    }
    tiles = [int(core_tile) for core_tile in dispatch.core_table()[:cores]]
    return thread_module.Ranks(line, tiles)


def _region_bytes(value):
    return int(value) if isinstance(value, int) else np.asarray(value).nbytes


def _region_initial(value):
    """The initial bytes as uint8, or None for a zeroed region."""
    if isinstance(value, int):
        return None
    return np.ascontiguousarray(value).view(np.uint8).reshape(-1)


def _host_array(value):
    """A shared region for the host fallbacks: uint32 words of its initial bytes."""
    size = _region_bytes(value)
    data = np.zeros(-(-size // 4) * 4, dtype=np.uint8)
    initial = _region_initial(value)
    if initial is not None:
        data[:size] = initial
    return data.view(np.uint32)


def _bank_regions(session, program, works):
    """{bank: _Regions} for the threads' own regions and the shared ones."""
    out = {}
    for bank in sorted({work.bank for work in works}):
        mine = [work for work in works if work.bank == bank]
        names = {name for work in mine for name in work.shared}
        shared = {
            name: value for name, value in program.shared.items() if name in names
        }
        out[bank] = _Regions(session, bank, mine, shared)
    return out


def _work_ring(dispatch, work):
    """The thread's ring (RingPlacement) and the kernel's end of it (tile, area)."""
    info = dispatch.work_info(work.cluster)
    return thread_module.work_ring(
        dispatch.lim_base,
        (dispatch.l2_x, dispatch.l2_y),
        info["lim_base"],
        (info["virtual_x"], info["virtual_y"]),
        work.cluster,
    )


def _pages(data, pages):
    """`data` (uint8) zero-padded to `pages` region pages, as [pages, words] uint32."""
    out = np.zeros(pages * thread_module.PAGE, dtype=np.uint8)
    out[: data.size] = data
    return out.view(np.uint32).reshape(pages, -1)


class _Regions:
    """The regions of one DRAM bank in one DRAM tensor whose pages of that bank hold
    them: page 0 the check line, then each thread's own regions (keys (cluster,
    name)), then the shared ones (("shared", name)). Written before the grid starts;
    read(key) only once the grid has stopped, from that bank's pages alone."""

    def __init__(self, session, bank, threads, shared):
        self.session = session
        self.check = thread_module.new_check_value()
        self.bank = bank
        entries = [
            ((thread.cluster, name), thread.region_bytes(name), thread.initial(name))
            for thread in threads
            for name in thread.regions
        ]
        entries += [
            (("shared", name), _region_bytes(value), _region_initial(value))
            for name, value in shared.items()
        ]
        self.placed, pages = {}, 1
        for key, size, _ in entries:
            self.placed[key] = thread_module.Placement(
                key[1], pages, -(-size // thread_module.PAGE), size
            )
            pages += self.placed[key].pages
        self.readable = False  # set by the run once the grid has stopped
        rows = np.zeros((DRAM_BANKS * pages, thread_module.PAGE // 4), dtype=np.uint32)
        mine = rows[self.bank :: DRAM_BANKS]
        mine[0, :2] = [self.check & abi.M32, self.check >> 32]
        for key, _, data in entries:
            placement = self.placed[key]
            if data is not None:
                mine[placement.page : placement.page + placement.pages] = _pages(
                    data, placement.pages
                )
        self.tensor = programs.words_tensor(session, rows)
        self.base = self.tensor.address
        if (
            self.base
            <= thread_module.IPI_ACK_DDR
            < self.base + pages * thread_module.PAGE
        ):
            raise StartError(
                f"the thread's regions at {self.base:#x} cover the IPI word"
            )

    def view(self, thread):
        """The regions `thread` sees, by name."""
        return _ThreadRegions(self, thread)

    def address(self, key):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.bank, self.base + self.placed[key].page * thread_module.PAGE

    def read(self, key):
        """The region's bytes, as uint8."""
        if not self.readable:
            raise RuntimeError("regions are read after run.stop()")
        placement, session = self.placed[key], self.session
        name = placement.name
        chunk_pages = (
            placement.pages
            if placement.pages * thread_module.PAGE <= REGION_WHOLE
            else REGION_CHUNK // thread_module.PAGE
        )
        spans = dram.bank_chunks(
            self.base,
            thread_module.PAGE,
            self.bank,
            placement.page,
            placement.pages,
            chunk_pages,
        )
        out = np.empty(placement.pages * thread_module.PAGE, np.uint8)
        for offset, span in zip(
            range(0, out.size, chunk_pages * thread_module.PAGE), spans
        ):
            (chunk,) = session._module.read_dram(session.device, [span])
            out[offset : offset + span[2]] = np.asarray(chunk, np.uint8)
            if len(spans) > 1:
                session.log(
                    f"region {name}: {offset + span[2]} of {out.size} bytes read"
                )
        return out[: placement.bytes].copy()


class _ThreadRegions:
    """One thread's regions in its bank's _Regions: its own, then the shared ones it
    names, in id order (table() gives the argument block's region table)."""

    def __init__(self, regions, thread):
        self.all = regions
        self.keys = {name: (thread.cluster, name) for name in thread.regions}
        self.keys |= {name: ("shared", name) for name in thread.shared}
        self.bank, self.base, self.check = regions.bank, regions.base, regions.check

    @property
    def readable(self):
        return self.all.readable

    @readable.setter
    def readable(self, value):
        self.all.readable = value

    def table(self):
        return [
            (self.address(name)[1], self.all.placed[key].bytes)
            for name, key in self.keys.items()
        ]

    def address(self, name):
        """(bank, bank address) of a region, for Tensix kernels."""
        return self.all.address(self.keys[name])

    def read(self, name):
        """The region's bytes, as uint8, after run.stop()."""
        return self.all.read(self.keys[name])
