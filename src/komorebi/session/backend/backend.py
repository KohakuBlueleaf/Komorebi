"""The card side of a Run: the native port, the device tensors, the readback. Made by
the session's load with the value it reads of the session (card.Card)."""

import re
import time
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from komorebi import abi
from komorebi.chain import calls as chain
from komorebi.device import extents, native
from komorebi.device.dram import DRAM_BANKS
from komorebi.errors import GridTimeout, L1BudgetError, ProgramError
from komorebi.islands import graph as islands
from komorebi.program import overlay
from komorebi.program import several as programs
from komorebi.program.addon import code as added
from komorebi.program.addon import program as added_program
from komorebi.program.checks import page_calls as island_names
from komorebi.program.description import messages_header
from komorebi.program.several import compute_config
from komorebi.session import kernels as kernel_module
from komorebi.session import process, sizes, tensor_bytes
from komorebi.session import state as device_state

MESSAGES_DRAM_BYTES = 4 << 30  # the most DRAM a dynamic handler's queues may take
STATE_DUMP_LINES = 8  # 64-byte lines of the kernel's state logged after a failure
PROBE_GAP_S = 0.02  # core 0's pass marker moves every 1,024 passes (about 33 us)
ABORT_GRID_S = 2.0  # an abort's wait for the grid after the stop flag, before the close
# a name the compiler gives a source file's own symbol (trisck.cc.4c8b3e06): the
# build's hash, which no part refers to
UNIT_SYMBOL = re.compile(r"\.cc\.[0-9a-f]+$")


class _TransferBuffer(NamedTuple):
    """The L1 of a run's transfers on every core: two chunks and a word slot."""

    address: int
    chunk: int
    bytes: int


class _Backend:
    """The card side of a Run: the native port, the device tensors, the readback."""

    def __init__(self, card, dispatch, mhz, trace):
        self.card = card  # what it reads of its session (backend.card.Card)
        self.device = card.device
        self.port = dispatch
        self.mhz = mhz
        self.trace_cap = (trace + 3) // 4 * 4 if trace else 0  # whole 64-byte pages
        self.io = []
        self.control = None
        self.control_addr = 0
        self.control_offset = 0
        self.trace_tensor = None
        self.state_tensor = None
        self.inbox = None  # the X280 ranks' messages to each core, in L1
        self.direct = None  # each core's direct messages' tables and slots, in L1
        self.stamps = None  # stamps=True: each core's reader and writer sums, in L1
        self.launch = None  # the launched program, kept for reset_core
        self.read = False
        self.close_failed = (
            ""  # the runtime's error message when an abort did not restore
        )
        self.aborted = None  # {words, cores}: what the kernel left at an abort
        self.transfer = None  # a _TransferBuffer: with the host channel, or threads
        self.apart_links = 0  # link cores holding the link area alone (_apart_links)
        self.link_kernels = 0  # link cores running the runtime's link kernels
        self.link_queue_at = None  # the put queue's offset from the link cores' base
        self.link_ring = True  # False: no chunks on link cores holding the slots too
        # a split card: (graph, dispatches, tags, [(first row, rows)], the islands whose
        # cores returned) of the one launch; `rows` this backend's island's
        self.split = None
        self.rows = None
        # a switching run's added programs, set by program.addon.program.pad_reserve
        self.switch_pad = None  # the pad's buffer index
        self.switch_loaded = 0  # programs loaded
        self.added_own = set()  # own overlay numbers added classes took
        self.added_entry = False  # the entry command's code is in the service overlay
        self.switch_args = False  # the load command's row read is built
        self.switch_reserve = 0  # L1 reserved for added programs

    def block(self, data, place=None, at=None):
        """A command's block (uint8 data) in DRAM: a copy at the same address of every
        bank, written while the grid runs; fence() before the command is sent. The
        tensor; free it after the command. `place(address)`: the data at that address.
        `at`: a bank address the caller holds (chain_records), its free() a no-op.
        """
        words = -(-len(data) // 64) * 16  # whole 64-byte lines: a core reads lines
        tensor = (
            self.card.empty((DRAM_BANKS, words), "uint32")
            if at is None
            else SimpleNamespace(address=at, free=lambda: None)
        )
        data = data if place is None else place(tensor.address)
        padded = np.zeros(4 * words, np.uint8)
        padded[: len(data)] = data
        self.card.module.write_dram(
            self.device, list(range(DRAM_BANKS)), tensor.address, padded
        )
        return tensor

    def fence(self):
        """Returns once every block written so far is in DRAM."""
        self.card.module.dram_barrier(self.device)

    def inspect(self, count):
        """While the kernel runs: the inspect block with its first `count` records, and
        the dispatcher's count of commands it gave up (LIM words, read by the runtime).
        """
        read = self.port.inspect(count)
        return {
            "block": np.asarray(read["block"]).ravel(),
            "given_up": int(read["given_up"]),
        }

    def _control_blocks(self, cores, grid, port_lines=None, area_bytes=0, bounds=None):
        """Zeroed control blocks, one shard per core, padded so that the block can
        start on a 64-byte line: the same L1 address on every core. `port_lines`: the
        KM_X280_TO line of each served core, written once the address is known.
        `area_bytes`: the overlay area after each block (written by load_overlays)."""
        abi_headers = abi.headers()
        words = (
            abi_headers["KM_CONTROL_BYTES"] + area_bytes + tensor_bytes.CONTROL_ALIGN
        ) // 4
        words += 0 if bounds is None else bounds[0].size
        self.control = programs.words_tensor(self.card, np.zeros((cores, words)), grid)
        base = self.control.address
        self.control_addr = (
            (base + tensor_bytes.CONTROL_ALIGN - 1)
            // tensor_bytes.CONTROL_ALIGN
            * tensor_bytes.CONTROL_ALIGN
        )
        self.control_offset = (self.control_addr - base) // 4
        kernel_module.check_bases({"control block": self.control_addr})
        self.control_rows = np.zeros((cores, words), dtype=np.uint32)
        self._island_rows(cores, grid[0])
        self._page_bounds_rows(bounds, area_bytes)
        if port_lines is not None:
            rows = self.control_rows
            to_offset = (
                self.control_offset
                + (abi_headers["KM_CONTROL_X280"] + abi_headers["KM_X280_TO"]) // 4
            )
            for core, (tile, box) in port_lines.lines.items():
                rows[core, to_offset + abi_headers["KM_X280_TO_TILE"]] = tile
                rows[core, to_offset + abi_headers["KM_X280_TO_BOX"]] = box
            self.control.write(rows)
        return self.control_addr

    def _page_bounds_rows(self, bounds, area_bytes):
        """KM_GUARD's list after the control block and overlay area, its address in
        the island area's table word (km_extent.hpp); its tensors kept alive by the run
        (an address freed and made again would not be the listed tensor)."""
        self.bounds_tensors = []
        if bounds is None or self.card.tensor_table:
            return
        words, self.bounds_tensors, left = bounds
        abi_headers = abi.headers()
        offset = (abi_headers["KM_CONTROL_BYTES"] + area_bytes) // 4
        rows = self.control_rows
        rows[
            :, self.control_offset + offset : self.control_offset + offset + words.size
        ] = words
        table = abi_headers["KM_CONTROL_ISLANDS"] + abi_headers["KM_ISLAND_TENSORS"]
        rows[:, self.control_offset + table // 4] = self.control_addr + 4 * offset
        said = extents.unchecked_says(left, self.card.page_bounds)
        if said:
            self.card.log(said)

    def _island_rows(self, cores, columns):
        """Every core's island area in control_rows: island i of the load's graph,
        else island 0 of 1 with every core a rank (its header, then its table, then
        the link area's and the tensor table's addresses); a split card's islands their
        own rows."""
        abi_headers = abi.headers()
        graph, here = self.card.island or (islands.one(cores, columns), 0)
        parts = [(here, 0, cores)]
        if self.split is not None:
            graph, _, _, rows, _ = self.split
            parts = [
                (i, first * columns, n * columns) for i, (first, n) in enumerate(rows)
            ]
        at = self.control_offset + abi_headers["KM_CONTROL_ISLANDS"] // 4
        line = abi_headers["KM_ISLAND_HEADER_BYTES"] // 4
        plan = self.card.link_plan
        plans = self.card.link_plans if self.split else None
        # the grid's panels: a whole island's release multicast (km::team_go)
        panels = self.card.nocs().panels if len(graph.islands) > 1 else None
        for here, base, count in parts:
            plan = plans[here] if plans else plan
            area = plan.area if plans else self.card.link_area
            header = [here, len(graph.islands), graph.first(here), graph.count(here)]
            header += [base, self.card.packet]
            header += [0]  # KM_ISLAND_LINKS: unused (the relay's, retired)
            mine = self.control_rows[base : base + count]
            if getattr(plan, "table", 0):  # the table in L1, the ports in the area
                header += [plan.table]
                mine[:, at + line : at + line + len(plan.ports)] = plan.ports
            else:  # no links: the table in the area, where the ports go
                table = np.frombuffer(graph.table(here), np.uint32)
                if line * 4 + table.nbytes > abi_headers["KM_ISLAND_LINK_AREA"]:
                    raise ValueError(f"{len(graph.islands)} islands without links")
                mine[:, at + line : at + line + len(table)] = table
            mine[:, at : at + len(header)] = header
            if plan is not None:  # two words 0, then the far link core 0's tile
                relay = at + abi_headers["KM_ISLAND_RELAY"]
                mine[:, relay : relay + 3] = plan.relay
                for core, tile in getattr(plan, "far_tiles", {}).items():
                    self.control_rows[core, relay + 2] = tile
            mine[:, at + abi_headers["KM_ISLAND_LINK_AREA"] // 4] = area
            mine[:, at + abi_headers["KM_ISLAND_TENSORS"] // 4] = self.card.tensor_table
            mine[:, at + abi_headers["KM_ISLAND_FORWARD"] // 4] = getattr(
                plan, "forward", 0
            )
            mine[:, at + abi_headers["KM_ISLAND_OWN"] // 4] = getattr(plan, "own", 0)
            if panels is not None:
                mine[:, at + abi_headers["KM_ISLAND_GO"] // 4] = (
                    kernel_module.go_layout(panels, columns, count, abi_headers)
                )

    def _direct_islands(self, program, cores, columns):
        """The direct areas of an island load (KM_DELIVER: a pair across islands goes
        by the link service): the pairs over the message ranks of every island, in
        flat order (each island's last `counters` ranks its counter tiles); this load's
        cores their rows."""
        messages = program.messages
        if "KM_DELIVER" not in program.defines:
            raise ValueError(
                "direct messages across islands need the deliveries (KM_DELIVER): a "
                "split card's dynamic load"
            )
        graph, here = self.card.island or (None, 0)
        parts = [(here, 0, cores)]
        if self.split is not None:
            graph, _, _, rows, _ = self.split
            parts = [
                (i, first * columns, n * columns) for i, (first, n) in enumerate(rows)
            ]
        held = [graph.count(i) - messages.counters for i in range(len(graph.islands))]
        flat = messages.direct_rows(sum(held), sum(held))
        out = np.zeros((cores, flat.shape[1]), np.uint32)
        for island, base, _ in parts:
            start = sum(held[:island])
            out[base : base + held[island]] = flat[start : start + held[island]]
        return out

    def _dynamic(self, program, cores, depth, base, grid, x280=None):
        """A dynamic handler's queues in DRAM (zero-filled by the cores) and its
        reader / writer and compute words; status lines in the kernel's LIM; with the
        X280 ranks' words `x280`, each core's inbox in L1 (zeroed)."""
        messages = program.messages
        if messages is None:
            return [], []
        if x280 is not None:
            constants, _ = messages_header()
            words = constants["kMessagesInbox"] * messages.message_bytes // 4
            self.inbox = programs.words_tensor(
                self.card, np.zeros((cores, words)), grid
            )
            self.io.append(self.inbox)
            x280 = x280 | {"inbox": self.inbox.address}
        abi_headers = abi.headers()
        messages.check(cores, MESSAGES_DRAM_BYTES, depth)
        page = 32 * messages.message_bytes  # kMessagesPageMessages messages a page
        queue = self.card.empty(
            (cores * messages.queue_messages // 32, page // 4), "int32"
        )
        self.io.append(queue)
        body = 0
        if messages.body_bytes:  # a page of body_bytes a queue position
            bodies = self.card.empty(
                (cores * messages.queue_messages, messages.body_bytes // 4), "int32"
            )
            self.io.append(bodies)
            body = bodies.address
        lines = base + abi_headers["KMD_LIM_MESSAGES"]
        direct = 0
        if messages.direct:  # each core's tables, credit lines and slots, in L1
            rows = messages.direct_rows(cores)
            if self.card.island is not None or self.split is not None:
                rows = self._direct_islands(program, cores, grid[0])
            self.direct = programs.words_tensor(self.card, rows, grid)
            self.io.append(self.direct)
            direct = self.direct.address
        stamps = 0
        if messages.stamps:  # km_messages.hpp kMessagesStamp*: cycles and counts
            words = messages_header()[0]["kMessagesStampWords"]
            zeros = np.zeros((cores, words), np.uint32)
            self.stamps = programs.words_tensor(self.card, zeros, grid)
            self.io.append(self.stamps)
            stamps = self.stamps.address
        reader_writer = messages.args(queue.address, lines, body, x280, direct, stamps)
        # another island's X280 ranks (island 0's, delivered): compute's alone
        elsewhere = getattr(program, "x280_elsewhere", 0)
        return reader_writer, messages.compute_args(cores, direct, elsewhere)

    def describe(self, *args, **options):
        """_describe's descriptor with the slots' buffer declared first, its size
        set once the overlays are built (_slot_buffer): what is built is launched."""
        self.link_cb, self.link_bytes = None, 0  # the link cores' buffer, its bytes
        self.link_kernels, self.link_queue_at, self.queue_after_slots = 0, None, False
        launch = self._describe(*args, **options)
        # below the highest index of the ranks' own: the link cores' buffer is above
        self.slot_index = kernel_module.slot_index(self.cb_indices)
        launch.cb(self._slot_cores(), self.slot_index, "uint32", 64, 64, first=True)
        return launch

    def _slot_cores(self):
        """The cores that hold slots: every rank (an island's link cores apart)."""
        grid = (self.port.grid_x, self.port.grid_y)
        # the link cores run no overlay: their put queue at their base on every island
        apart = self.apart_links or self.link_kernels
        if apart:
            return programs.rectangles(0, grid[0] * grid[1] - apart, grid[0])
        return [(0, 0, grid[0] - 1, grid[1] - 1)]

    def _describe(
        self,
        program,
        sources,
        tag,
        depth,
        wait,
        idle,
        trace,
        port_lines=None,
        placed=None,
        x280=None,
    ):
        """The program descriptor for every core, with the start-up arguments; with
        `placed` (a programs.Plan) the runtime's kernels, per group of cores; `x280`:
        the X280 ranks' words of a dynamic program."""
        abi_headers = abi.headers()
        port = self.port
        grid = (port.grid_x, port.grid_y)
        cores = grid[0] * grid[1]
        self.slots_most = kernel_module.slots_most(program, placed)
        # the service overlay's too, and the records of slots past 4
        self.area_bytes = overlay.area_bytes(self.slots_most)
        island_names.check(placed.programs if placed else [program], self.card.log)
        bounds = self.card.page_bounds_list(placed.programs if placed else [program])
        control = self._control_blocks(cores, grid, port_lines, self.area_bytes, bounds)
        table = np.zeros((1, abi_headers["KM_CORES_MAX"]), dtype=np.uint64)
        table[0] = port.core_table()
        cores_tensor = programs.words_tensor(self.card, table)
        self.io = [self.control, cores_tensor]
        if self.card.channel or program.threads:
            # also the threads' copies; a prebuild's as the card's
            chunk = self._transfer_chunk(program, placed, cores)
            self.transfer = self._transfer_buffer(cores, grid, chunk)
        if self.trace_cap:
            self.trace_tensor = programs.words_tensor(
                self.card,
                np.zeros((cores, self.trace_cap * abi_headers["kTraceWords"])),
            )
            self.io.append(self.trace_tensor)
        base = port.lim_base
        groups = abi.column_groups(grid, cores)
        common = {
            "control": control,
            "cores": cores_tensor.address,
            "dispatcher_coordinates": port.l2_x | port.l2_y << 16,
            "ack_base": base + abi_headers["KMD_LIM_ACKS"],
            "ready_base": base + abi_headers["KMD_LIM_READY"],
            "tag": tag,
            "depth": depth,
            "workers": cores,
            "row": grid[0],
            "wait_limit": abi.limit_units(wait, self.mhz),
            "idle_limit": abi.limit_units(idle, self.mhz) if idle else 0,  # 0: none
            "trace": self.trace_tensor.address if self.trace_tensor else 0,
            "trace_cap": self.trace_cap,
        }
        messages_reader_writer_args, messages_compute_args = self._dynamic(
            program, cores, depth, base, grid, x280
        )
        owners = [(0, common, base)] * cores  # per core: its island's first core
        if self.split is not None:  # each island's cores hello its own dispatcher
            owners, groups = self._split_owners(common, grid)
        reader_args, writer_args, compute_args = [], [], []
        for core in range(cores):
            x, y = core % grid[0], core // grid[0]
            leader, rank, peers, slot = groups[core]
            first_core, mine, lim = owners[core]
            start = abi.start_args(
                mine
                | {
                    "me": core,
                    "mailbox": (
                        lim + abi_headers["KMD_LIM_MAILBOX"]
                        if core == first_core
                        else 0
                    ),
                    "leader": leader,
                    "rank": rank,
                    "peers": peers,
                    "ready_slot": slot,
                }
            )
            reader, writer, compute = kernel_module._core_args(program, placed, core)
            reader_args.append(
                (x, y, start + messages_reader_writer_args + list(reader))
            )
            writer_args.append(
                (x, y, start + messages_reader_writer_args + list(writer))
            )
            limits = [common["wait_limit"], common["idle_limit"]]
            compute_args.append(
                (x, y, [control, core, *limits, *messages_compute_args, *compute])
            )
        launch = self.card.module.Launch(self.device)
        if placed is not None:
            args = (reader_args, writer_args, compute_args)
            self._describe_groups(launch, placed, sources, args, grid)
            return launch
        all_cores = [(0, 0, grid[0] - 1, grid[1] - 1)]
        include = [str(abi.headers().abi.parents[2])]
        defines = [(name, str(value)) for name, value in program.defines.items()]
        defines += self._one_island()
        defines += kernel_module.own_slot_path(program)
        defines += overlay.slots_define(self.slots_most)
        if program.messages is not None:
            defines += program.messages.defines()
        ranks, links = kernel_module._link_kernels(
            self.card.island, program, self.split, cores
        )
        if "KM_PAGE_LINKS" in program.defines and not links:
            defines.append(("KM_LINK_SERVICES", "1"))  # its link cores run it too
        held = programs.rectangles(0, ranks, grid[0]) if links else all_cores
        apart = self._apart_links(links)
        self.cb_indices = set()  # every buffer index of the launch, the slots' apart
        for index, fmt, page, pages in self._cbs(program.buffers):
            launch.cb(held, index, fmt, page, page * pages)
            self.cb_indices.add(index)
        if "KM_PAGE_LINKS" in program.defines and not links:
            # its link cores' put queue and the least chunks a put ring takes, on them
            # alone (the program's kernels there); no room: no ring, no put batch
            packet = self.card.packet or 0
            chunk = min(
                self._link_chunk(program, cores), kernel_module.link_chunk_least(packet)
            )
            chunk = chunk if self.link_ring else 0
            defines += self._link_queue(launch, program, grid, chunk)
            if chunk:
                defines.append(("KM_LINK_CHUNK", str(chunk)))
                self.card.log(f"link cores: their own chunks, {chunk:,} B a CPU")
            self.link_queue_at = (
                kernel_module._cbs_end(self._cbs(program.buffers)),
                chunk,
            )
            self.queue_after_slots = True  # the slots' bytes added (_slot_buffer)
        if apart:  # the link cores: their link area first, then the tokens
            self._link_area(launch, ranks, links, grid, apart)
            served = programs.rectangles(ranks, links, grid[0])
            for index, fmt, page, pages in self._runtime_cbs():
                launch.cb(served, index, fmt, page, page * pages)
        kernels = [("reader", reader_args, 0, {}), ("writer", writer_args, 1, {})]
        kernels.append(("compute", compute_args, 2, compute_config(program.compute)))
        for kernel, args, processor, config in kernels:
            launch.kernel(
                str(sources[kernel]),
                held,
                processor,
                program.kernel_compile_args(kernel),
                includes=include,
                args=args[:ranks],
                **kernel_module.kernel_build(program, kernel, defines),
                **config,
            )
        if links:  # the island's link cores: the runtime's link kernels, its services
            served = programs.rectangles(ranks, links, grid[0])
            link_defines = [
                (name, str(value))
                for name, value in program.defines.items()
                if name in kernel_module.LINK_DEFINES
            ] + [("KM_LINK_SERVICES", "1")]
            chunk = self._link_chunk(program, cores, apart)
            # the link cores' own L1, on them alone: the put queue, the chunks
            link_defines += self._link_queue(launch, program, grid, chunk)
            self.link_kernels = links
            below = [(0, "uint32", apart, 1), *self._runtime_cbs()]
            self.link_queue_at = (kernel_module._cbs_end(below), chunk)
            if chunk:
                link_defines.append(("KM_LINK_CHUNK", str(chunk)))
                self.card.log(f"link cores: their own chunks, {chunk:,} B a CPU")
            kernel_module.check_link_defines(program.name, link_defines, [program])
            for kernel, args, processor, _ in kernels:
                launch.kernel(
                    str(kernel_module._link_kernel_path(kernel)),
                    served,
                    processor,
                    [],
                    includes=include,
                    args=args[ranks:],
                    defines=link_defines,
                    **(compute_config({}) if kernel == "compute" else {}),
                )
        return launch

    def _one_island(self):
        """KM_ONE_ISLAND for a load that is no island of several (a card of its own,
        not split): its kernels take the island table's one entry as known
        (km_island.hpp), the code of several islands left out."""
        if self.card.island is not None or self.split is not None:
            return []
        return [("KM_ONE_ISLAND", "1")]

    def _apart_links(self, links):
        """The bytes of the link area at the link cores' buffers' base when
        the load's link plan keeps them there (its link cores run their own kernels,
        `links` of them), else 0; the rank cores' buffers are then launched apart."""
        plan = self.card.link_plan
        if plan is None or getattr(plan, "apart", None) is None:
            return 0
        if not links:
            raise ProgramError(
                "the link plan keeps the link area on the link cores alone, but they "
                "run the program's kernels"
            )
        self.apart_links = links
        return 4 * len(plan.apart[1])

    def _link_area(self, launch, ranks, links, grid, size):
        """The link area (`size` bytes) as the first buffer of the `links`
        link cores after the `ranks` ranks: at their buffers' base on every island."""
        runtime = {cb[0] for cb in self._runtime_cbs()} | {abi.headers()["KM_LINK_CB"]}
        index = min(set(range(32)) - runtime)
        launch.cb(programs.rectangles(ranks, links, grid[0]), index, "uint32", 64, size)

    def _split_owners(self, common, grid):
        """A split card's per-core (island's first core, start-up words, LIM base) and
        start-up groups: each island's column leaders report to its own dispatcher."""
        abi_headers = abi.headers()
        _, ports, tags, rows, _ = self.split
        owners, groups = [], {}
        for (first, count), port, tag in zip(rows, ports, tags):
            lim = port.lim_base
            mine = common | {
                "dispatcher_coordinates": port.l2_x | port.l2_y << 16,
                "ack_base": lim + abi_headers["KMD_LIM_ACKS"],
                "ready_base": lim + abi_headers["KMD_LIM_READY"],
                "tag": tag,
            }
            low = first * grid[0]
            owners += [(low, mine, lim)] * (count * grid[0])
            local = abi.column_groups((grid[0], count), count * grid[0])
            for core, (leader, rank, peers, slot) in local.items():
                groups[low + core] = (low + leader, rank, peers, slot)
        return owners, groups

    def _transfer_chunk(self, program, placed, cores):
        """The chunk of the run's transfer buffer (_transfer_chunk_of)."""
        held = placed.programs if placed else [program]
        switch = bool(placed and placed.switch)
        card = self.card
        return tensor_bytes._transfer_chunk_of(
            card.largest_page(), card.l1, card.transfer_chunk, held, switch,
            self.trace_cap, self.control.nbytes // cores,
        )  # fmt: skip

    def _link_core_rects(self, grid):
        """The rectangles of this load's link cores (two cards: its island's; a split
        card: every island's), [] for none."""
        if self.split is not None:
            graph, mine = self.split[0], range(len(self.split[0].islands))
        elif self.card.island is not None:
            graph, here = self.card.island
            mine = [here]
        else:
            return []
        rects = []
        for island in mine:
            cores = graph.link_cores[island]
            if cores:
                rects += programs.rectangles(cores[0], len(cores), grid[0])
        return rects

    def _link_queue(self, launch, program, grid, chunk):
        """The link cores' own L1 (KM_LINK_CB, on them alone): the put queue, then
        two chunks of `chunk` bytes (0: none); the kernels' defines that say so."""
        rects = self._link_core_rects(grid)
        if not rects:
            return []
        h = abi.headers()
        index = h["KM_LINK_CB"]
        if index in self.cb_indices:
            raise ValueError(
                f"{program.name}: buffer index {index} is the link cores' own"
            )
        size = (
            h["KM_QUEUE_BYTES"]
            + sizes.far_bytes(program.defines)
            + sizes.forward_bytes(program.defines)
            + sizes.deliver_bytes(program.defines)
            + 2 * chunk
        )
        launch.cb(rects, index, "uint32", size, size)
        self.link_cb, self.link_bytes = index, size
        return [("KM_LINK_QUEUE", "1")]

    def _link_chunk(self, program, cores, apart=0):
        """A link CPU's own chunk (bytes; 0: none): at most LINK_CHUNK_MOST, at least
        link_chunk_least, whole 64 KiB, two in what a link core has left after the
        program's buffers (unless `apart`), the transfer buffer, what the session keeps
        and the put queue."""
        buffers = 0
        if not apart:  # else launched on the ranks only (buffers per core range)
            buffers = sum(
                buffer.page_bytes * buffer.pages for buffer in program.buffers
            )
        buffers += 2 * abi.headers()["kTokenBytes"] + self.trace_cap * 16
        buffers += self.transfer.bytes if self.transfer is not None else 0
        spare = (
            self.card.l1()[1]
            - buffers
            - self.control.nbytes // cores
            - self._held_bytes()
            - apart
            - abi.headers()["KM_QUEUE_BYTES"]
            - sizes.far_bytes(program.defines)
            - sizes.forward_bytes(program.defines)
            - sizes.deliver_bytes(program.defines)
            - kernel_module.LINK_CHUNK_MARGIN
        )
        chunk = min(
            kernel_module.LINK_CHUNK_MOST, max(spare, 0) // 2 // (64 << 10) * (64 << 10)
        )
        least = kernel_module.link_chunk_least(self.card.packet or 0)
        return chunk if chunk >= least else 0  # else no ring: the transfer chunk's

    def _transfer_buffer(self, cores, grid, chunk):
        """The L1 every core moves a transfer's chunks through (a sharded tensor)."""
        size = 2 * chunk + abi.headers()["KM_TRANSFER_WORD_BYTES"]
        tensor = self.card.empty(
            (cores, (size + sizes.NOC_ALIGN) // 4), "uint32", shard_grid=grid
        )
        self.io.append(tensor)
        address = -(-tensor.address // sizes.NOC_ALIGN) * sizes.NOC_ALIGN
        return _TransferBuffer(address, chunk, size + sizes.NOC_ALIGN)

    def _held_rows(self):
        """The L1 budget table's rows of what the session's owner keeps at the top of
        every core (two cards: the link area, the tensor table) and of
        what the session already holds there (l1_taken)."""
        return list(self.card.l1_held()) + self.card.l1_taken()

    def _taken_bytes(self):
        """The bytes of the session's own L1 held from before this load."""
        return sum(size for _, size in self.card.l1_taken())

    def release_l1(self):
        """A finished run's L1 tensors (its host transfer buffer, control blocks and
        trace, all read back) given back, so a later load of the session has them."""
        self.io = [tensor for tensor in self.io if tensor.in_dram]
        self.control = self.trace_tensor = None
        self.bounds_tensors = []

    def _held_bytes(self):
        """The bytes of _held_rows."""
        return sum(size for _, size in self._held_rows())

    def _cbs(self, buffers):
        """(index, format, page bytes, pages) of `buffers` and the runtime's own."""
        return programs.program_cbs(buffers) + self._runtime_cbs()

    def _runtime_cbs(self):
        abi_headers = abi.headers()
        token_bytes = abi_headers["kTokenBytes"]
        cbs = [
            (abi_headers["kCbTokenCompute"], "uint32", token_bytes, 1),
            (abi_headers["kCbTokenWriter"], "uint32", token_bytes, 1),
        ]
        if self.trace_cap:
            cbs.append((abi_headers["kCbTrace"], "uint32", self.trace_cap * 16, 1))
        return cbs

    def _switch_layouts(self, placed):
        """In a run whose groups switch: each program's buffers placed after the
        runtime's own ({field: layout}), and the end of the largest."""
        _, kernel_free, l1_bytes = self.card.module.kernel_buffer(self.device)
        _, start = programs.buffer_layout(self._runtime_cbs(), l1_bytes - kernel_free)
        self.switch_layouts, end = {}, start
        for program in placed.programs:
            # a group's buffers are launched in index order (joint_buffers)
            ordered = sorted(program.buffers, key=lambda buffer: buffer.index)
            layout, last = programs.buffer_layout(programs.program_cbs(ordered), start)
            self.switch_layouts[program.program_field] = layout
            end = max(end, last)
        self.switch_base = l1_bytes - kernel_free
        self.switch_start, self.switch_shift = start, 0  # an added program's start
        return end

    def _describe_groups(self, launch, placed, sources, args, grid):
        """The runtime's kernels and each group's buffers on its cores: a group's
        kernels are built apart (KM_PROGRAM_GROUP), with its buffers and options."""
        include = [str(abi.headers().abi.parents[2])]
        free = self.card.l1()[1]
        self.group_buffers = []
        self.cb_indices = {  # a switch may give any core any program's buffers
            buffer.index for program in placed.programs for buffer in program.buffers
        }
        island, ranks, links = self._placed_links(placed, grid)
        apart = self._apart_links(links)
        if apart:  # the link cores' link area first, at their buffers' base
            self._link_area(launch, ranks, links, grid, apart)
        queue = []  # the link cores' put queue, when a program puts across
        if any("KM_PAGE_LINKS" in program.defines for program in placed.programs):
            queue = self._link_queue(launch, placed.programs[0], grid, 0)
        align = programs.CB_ALIGN
        if placed.switch:  # the runtime's buffers first, then room for the largest
            reserve = -(-self._switch_layouts(placed) // align) * align
            reserve = added_program.pad_reserve(self, placed, reserve)
            self.switch_reserve = reserve - self.switch_base
        for number, group in enumerate(placed.groups):
            held = [placed.programs[index] for index in group.programs]
            buffers = programs.joint_buffers(held, free)
            self.group_buffers.append(buffers)
            cores = programs.rectangles(group.first, group.count, grid[0])
            cbs = self._cbs(buffers)
            if placed.switch:
                cbs = self._runtime_cbs() + programs.program_cbs(buffers)
                _, end = programs.buffer_layout(cbs, self.switch_base)
                pad = reserve - -(-end // align) * align
                if pad > 0:  # one index on every group: an add resizes it
                    cbs.append((self.switch_pad, "uint32", pad, 1))
            for index, fmt, page, pages in cbs:
                launch.cb(cores, index, fmt, page, page * pages)
                self.cb_indices.add(index)
            cb_in, cb_out = programs.start_buffers(buffers)
            defines = [
                ("KM_PROGRAM_GROUP", str(number)),
                ("KM_PROGRAM_CB_IN", str(cb_in)),
                ("KM_PROGRAM_CB_OUT", str(cb_out)),
                *queue,
            ]
            if any(b.index == chain.words()["kChainBuffer"] for b in buffers):
                defines.append(("KM_PROGRAM_CHAIN", "1"))  # the reader zeroes its words
            if placed.switch:  # the kernels do not depend on the group's buffers
                defines = [
                    ("KM_PROGRAM_GROUP", str(number)),
                    ("KM_PROGRAM_SWITCH", "1"),
                    *queue,
                ]
                if self.switch_args:  # the row read: pay for use
                    defines.append(("KM_ADDED_ARGS", "1"))
                if self.added_entry:  # own overlays named ahead
                    defines.append(("KM_ADDED_ENTRY", "1"))
            defines += island
            defines += self._one_island()
            defines += kernel_module.added_path(placed.programs)
            defines += kernel_module.guard_path(placed.programs)
            defines.append(("KM_OVERLAY_OWN", "1"))  # placed: every program's parts
            defines += overlay.slots_define(self.slots_most)
            first = held[0] if held else None
            if placed.switch and first is None:
                first = placed.programs[0]  # one build for every group's kernels
            mine = range(group.first, group.first + group.count)
            for processor, kernel in enumerate(("reader", "writer", "compute")):
                config = (
                    compute_config(first.compute if first else {})
                    if kernel == "compute"
                    else {}
                )
                build = (
                    kernel_module.kernel_build(first, kernel, defines)
                    if first
                    else {"defines": defines}
                )
                launch.kernel(
                    str(sources[kernel]),
                    cores,
                    processor,
                    [],
                    includes=include,
                    args=[args[processor][core] for core in mine],
                    **build,
                    **config,
                )
        if links:  # the island's link cores: the runtime's link kernels, its services
            self._link_cores(launch, placed.programs, args, ranks, links, grid, queue)
            self.link_kernels = links
            below = kernel_module._cbs_end([(0, "uint32", apart, 1)]) if apart else 0
            self.link_queue_at = (below, 0) if queue else None  # after the area

    def _placed_links(self, placed, grid):
        """A plan's island defines (those the island load gave its programs), and its
        (ranks, link cores) as _link_kernels: the plan covers the ranks only."""
        first = placed.programs[0]
        island = [
            (name, str(value))
            for name, value in first.defines.items()
            if name in kernel_module.ISLAND_DEFINES
        ]
        cores = grid[0] * grid[1]
        ranks, links = kernel_module._link_kernels(
            self.card.island, first, self.split, cores
        )
        if "KM_PAGE_LINKS" in first.defines and not links:
            island.append(("KM_LINK_SERVICES", "1"))  # its link cores run it too
        if placed.cores != ranks:
            raise ProgramError(
                f"a plan over {placed.cores} cores on an island of {ranks} ranks"
            )
        return island, ranks, links

    def _link_cores(self, launch, placed, args, ranks, links, grid, queue=()):
        """The runtime's link kernels on the cores after the `ranks` ranks, with the
        link defines of every program `placed` and `queue` (the put queue's, which the
        ranks' put lines need served), and the runtime's buffers (its tokens)."""
        include = [str(abi.headers().abi.parents[2])]
        served = programs.rectangles(ranks, links, grid[0])
        for index, fmt, page, pages in self._runtime_cbs():
            launch.cb(served, index, fmt, page, page * pages)
            self.cb_indices.add(index)
        named = {}
        for program in placed:
            named |= {
                k: str(v)
                for k, v in program.defines.items()
                if k in kernel_module.LINK_DEFINES
            }
        link_defines = [*named.items(), ("KM_LINK_SERVICES", "1"), *queue]
        kernel_module.check_link_defines(placed[0].name, link_defines, placed)
        for processor, kernel in enumerate(("reader", "writer", "compute")):
            launch.kernel(
                str(kernel_module._link_kernel_path(kernel)),
                served,
                processor,
                [],
                includes=include,
                args=args[processor][ranks:],
                defines=link_defines,
                **(compute_config({}) if kernel == "compute" else {}),
            )

    def _slot_buffer(self, descriptor, slot_bytes):
        """Every core's slots as the launch's first circular buffer, at the L1 base,
        so they lie below 1 MB and the buffers above them; returns their address."""
        base = sizes.L1_BYTES - self.card.l1()[1]
        address, size, _ = kernel_module.slot_area(
            base, slot_bytes, self.cb_indices, self.card.slow_code
        )
        index = self.slot_index  # declared by describe
        descriptor.resize_cb(index, size, size)
        if self.queue_after_slots and self.link_queue_at is not None:
            self.link_queue_at = (self.link_queue_at[0] + size, self.link_queue_at[1])
        # a switch's buffers of every program field move up with the rest
        if getattr(self, "switch_layouts", None):
            for layout in self.switch_layouts.values():
                for key, (at, *rest) in layout.items():
                    layout[key] = (at + size, *rest)
            self.switch_shift = size
        self.card.log(f"slots: {size:,} B from {address:#x} (buffer {index})")
        return address

    def load_overlays(self, program, descriptor):
        """Builds the program's overlays and the service overlay against the kernels
        just built, puts each part in every DRAM bank, places the slots in L1 (refused,
        with the L1 budget table, when they do not fit), writes each overlay area."""
        card, abi_headers = self.card, abi.headers()
        started = time.perf_counter()
        recipes = {}
        for source, builds in descriptor.recipes():
            if kernel_module._is_link_kernel(
                source
            ):  # the link cores': no overlay runs there
                continue
            for row in builds:
                recipe = overlay.Recipe.from_native(row)
                recipes[recipe.target] = recipe
        built = overlay.build_overlays(
            [*program.overlays.values(), overlay.service()],
            program.overlay_slots,
            recipes,
            native.cache_directory() / "overlays",
            card.log,
            self.slots_most,
        )
        rows = card.l1_rows(program, self.trace_cap) + sizes.link_rows(self)
        built = kernel_module.fit_slots(
            built,
            kernel_module.own_slot_declared(program),
            card.l1()[1],
            rows,
            card.log,
        )
        program.overlays_built = built
        rows[2:2] = kernel_module.slot_rows(built.slots, built.service, built.why)
        table, fits = overlay.budget(card.l1()[1], rows)
        card.log(f"L1 budget of a core:\n{table}")
        if not fits:
            refused = card.resize(program.sizes.total, l1_failed=True)
            raise L1BudgetError(
                program.name, table, self._held_bytes(), refused, self._taken_bytes()
            )
        image_rows, placed = overlay.dram_rows(built, getattr(self, "reserve", 0))
        image = programs.words_tensor(card, image_rows)
        slot_base = self._slot_buffer(descriptor, built.slot_bytes)
        self.io.append(image)
        ranks = len(self.control_rows) - self.apart_links  # link cores apart: no slots
        self.room = added.room(
            self,
            image,
            len(built.images),
            [added.group(0, ranks, recipes, built)],
            [program],
        )
        added.build_ahead(card, self.room, [program])  # Program.ahead's parts
        area = overlay.area_words(
            built,
            placed,
            image.address,
            image.aligned_page_bytes,
            slot_base,
            self.slots_most,
        )
        at = self.control_offset + abi_headers["KM_CONTROL_BYTES"] // 4
        self.control_rows[:, at : at + len(area)] = area
        if self.apart_links:  # no slots there: no overlay to load
            self.control_rows[-self.apart_links :, at : at + len(area)] = 0
        self.control.write(self.control_rows)
        card.log(
            f"overlays {[*program.overlays, 'service']} built and placed: slots from "
            f"{slot_base:#x}, {image.nbytes:,} B of DRAM "
            f"({time.perf_counter() - started:.2f} s)"
        )

    def _switch_line(self, area, placed, group):
        """A group's switch line in its overlay area: the program field it holds and
        its buffers (none for a group without one); every number in the table."""
        c = overlay.constants()
        area[c["kOverlayCount"]] = c["kOverlaysMax"]
        line = c["kOverlaySwitch"] // 4
        area[line + c["kSwitchHolding"]] = c["kSwitchNone"]
        area[line + c["kSwitchMask"]] = 0
        if group.programs:
            program_field = group.programs[0]
            area[line + c["kSwitchHolding"]] = program_field
            area[line + c["kSwitchMask"]] = sum(
                1 << index for index in self.switch_layouts[program_field]
            )

    def _switch_records(self, placed, image, placed_rows, symbols):
        """Each program's record in DRAM (its table entries and buffers), and why a
        group cannot switch, if the groups' kernels differ: self.switch for the Run."""
        c = overlay.constants()
        records = {}
        for program in placed.programs:
            number = c["kOverlayService"] - 1 - program.program_field
            table = []
            for part in overlay.PARTS:
                row, size = placed_rows.get((number, part), (None, 0))
                address = image.address + row * image.aligned_page_bytes if size else 0
                table.append((address, size))
            words = programs.record_words(
                program.program_field, self.switch_layouts[program.program_field], table
            )
            records[(program.program_field, "record")] = words.tobytes()
        spare = added_program.spare_fields(placed)  # rows for added classes' records
        empty = bytes(c["kRecordBytes"])
        records |= {(field, "record"): empty for field in spare}
        rows, placed_records = overlay.dram_rows(overlay.Built(records, {}))
        tensor = programs.words_tensor(self.card, rows)
        self.io.append(tensor)
        refused = None
        differ = sorted(
            (target, name)
            for each in symbols[1:]
            for target, names in each.items()
            for name in names.keys() | symbols[0].get(target, {}).keys()
            if names.get(name) != symbols[0].get(target, {}).get(name)
        )
        if differ:
            refused = (
                "the groups' runtime kernels differ (compute options or opt levels; "
                f"{differ[0][0]}'s {differ[0][1]}, {len(differ)} symbols), so a "
                "program's parts would not fit another group's kernels"
            )
        addresses = {
            key[0]: tensor.address + row * tensor.aligned_page_bytes
            for key, (row, _) in placed_records.items()
        }
        self.switch = {
            "records": {f: a for f, a in addresses.items() if f not in spare},
            # an added class's record goes in its row (run.addon(program))
            "spare": {f: (addresses[f], placed_records[f, "record"][0]) for f in spare},
            "tensor": tensor,
            "refused": refused,
            # an island's part of a plan: programs none of its groups holds, whose parts
            # were not built here (a part is built against its group's kernels' buffers)
            "unheld": {
                program.program_field: program.name
                for index, program in enumerate(placed.programs)
                if not any(index in group.programs for group in placed.groups)
            },
        }

    def load_programs(self, placed, descriptor):
        """Builds each program as an overlay against its group's runtime kernels, puts
        the parts in every DRAM bank, places the slots (refused, with the L1 budget
        table, when a group's do not fit) and writes each group's overlay area."""
        card, abi_headers = self.card, abi.headers()
        started = time.perf_counter()
        by_group = kernel_module._group_recipes(
            descriptor.recipes(), len(placed.groups)
        )
        declared = placed.programs[0].overlay_slots
        built, symbols = [], []
        for number, group in enumerate(placed.groups):
            recipes = by_group[number]
            if placed.switch:  # a part runs on any group's kernels: the same globals
                symbols.append(
                    {
                        target: {
                            name: value
                            for name, (value, _, binding) in overlay.elf_symbols(
                                recipe, recipe.elf
                            ).items()
                            if binding != "LOCAL" and not UNIT_SYMBOL.search(name)
                        }
                        for target, recipe in recipes.items()
                    }
                )
            held = [placed.programs[index] for index in group.programs]
            # a group that may switch program holds every program's own overlays
            owners = placed.programs if placed.switch else held
            fits = False
            while not fits:  # a merged program over L1 places a source as an overlay
                declared = placed.programs[0].overlay_slots
                made = overlay.build_overlays(
                    [
                        *(
                            part
                            for program in held
                            for part in programs.program_overlays(
                                program, placed.switch, len(placed.programs) == 1
                            )
                        ),
                        *(
                            each
                            for owner in owners
                            for each in programs.own_overlays(owner)
                        ),
                        *added_program.ahead_overlays(self, placed, number, recipes),
                        overlay.service(),
                    ],
                    declared,
                    recipes,
                    native.cache_directory() / "overlays",
                    card.log,
                    self.slots_most,
                )
                buffers = sum(
                    buffer.page_bytes * buffer.pages
                    for buffer in self.group_buffers[number]
                )
                rows = card.l1_rows(
                    placed.programs[0], self.trace_cap, placed, buffers=buffers
                ) + sizes.link_rows(self)
                if not placed.switch:  # a switch's slots are common: sized below
                    made = kernel_module.fit_slots(
                        made, declared, card.l1()[1], rows, card.log
                    )
                rows[2:2] = kernel_module.slot_rows(made.slots, made.service, made.why)
                table, fits = overlay.budget(card.l1()[1], rows)
                names = ", ".join(program.name for program in held)
                card.log(f"L1 budget of a core of {names}:\n{table}")
                spills = [p.spill for p in held if hasattr(p, "spill")]
                if not fits and not any(spill(card.log) for spill in spills):
                    break
            for program in held:
                program.overlays_built = made
            built.append(made)
            if not fits:
                refused = card.resize(placed.programs[0].sizes.total, True)
                raise L1BudgetError(
                    names, table, self._held_bytes(), refused, self._taken_bytes()
                )
        images, stored = {}, []
        service = overlay.constants()["kOverlayService"]
        for made in built:
            mine = {}
            for key, image in made.images.items():
                at = key
                if key[0] == service or key in images:  # a copy for each build of it
                    same = [k for k, v in images.items() if k[:2] == key and v == image]
                    at = same[0] if same else (*key, len(images))
                images[at] = image
                mine[key] = at
            stored.append(mine)
        if placed.switch:  # one set of slots, every program's parts fit any core's
            common = kernel_module._common_slots(
                placed, declared, built, self.slots_most
            )
            rows = card.l1_rows(
                placed.programs[0], self.trace_cap, placed
            ) + sizes.link_rows(self)
            # the runtime's buffers included
            rows[0] = ("circular buffers (the largest program)", self.switch_reserve)
            common = kernel_module.fit_slots(
                common, declared, card.l1()[1], rows, card.log
            )
            kernel_module._own_slots_kept(placed, common)
            built = [
                overlay.Built(made.images, common.slots, common.service)
                for made in built
            ]
            rows[2:2] = kernel_module.slot_rows(
                common.slots, common.service, common.why
            )
            added_program.grow_area(self, descriptor, rows, placed)
            table, fits = overlay.budget(card.l1()[1], rows)
            card.log(f"L1 budget of a core:\n{table}")
            if not fits:
                refused = card.resize(placed.programs[0].sizes.total, True)
                raise L1BudgetError(
                    ", ".join(p.name for p in placed.programs),
                    table,
                    self._held_bytes(),
                    refused,
                    self._taken_bytes(),
                )
        reserve = getattr(
            self, "reserve", 0
        )  # the ranks' room, link cores apart or not
        if reserve:  # at least the rows the classes named ahead take, a part a row
            width = max(len(image) for image in images.values())
            reserve = max(reserve, added_program.ahead_parts(placed) * width)
        image_rows, placed_rows = overlay.dram_rows(overlay.Built(images, {}), reserve)
        image = programs.words_tensor(card, image_rows)
        slot_base = self._slot_buffer(
            descriptor, max(made.slot_bytes for made in built)
        )
        self.io.append(image)
        self.room = added.room(
            self,
            image,
            len(images),
            [
                added.group(group.first, group.count, by_group[number], made)
                for number, (group, made) in enumerate(
                    zip(placed.groups, built, strict=True)
                )
            ],
            placed.programs,
        )
        added.build_ahead(card, self.room, placed.programs)  # Program.ahead's parts
        added_program.build_ahead(card, self, self.room, placed)  # classes ahead
        at = self.control_offset + abi_headers["KM_CONTROL_BYTES"] // 4
        for group, made, keys in zip(placed.groups, built, stored, strict=True):
            mine = {key: placed_rows[at] for key, at in keys.items()}
            area = overlay.area_words(
                made,
                mine,
                image.address,
                image.aligned_page_bytes,
                slot_base,
                self.slots_most,
            )
            if placed.switch:
                self._switch_line(area, placed, group)
            rows = slice(group.first, group.first + group.count)
            self.control_rows[rows, at : at + len(area)] = area
        self.control.write(self.control_rows)
        if placed.switch:
            self._switch_records(placed, image, placed_rows, symbols)
        card.log(
            f"programs {[program.name for program in placed.programs]} built and "
            f"placed: slots from {slot_base:#x}, {image.nbytes:,} B of DRAM "
            f"({time.perf_counter() - started:.2f} s)"
        )

    def finish(self, timeout):
        """Waits for the kernel to finish, stops it, then closes the X280 runtime."""
        try:
            return self.port.finish(timeout)
        finally:
            self.port.close()
            self.card.log("x280 runtime closed")

    def abort(self):
        """The failure path: the stop flag to every core, a timed wait for the grid,
        then the runtime's close, so core 0 is not held behind a grid that runs on.
        Returns whether cluster 0 was restored; `aborted` holds what the kernel left."""
        try:
            self.stop_grid()
            back = self._wait_grid(ABORT_GRID_S)
            self.card.log("grid returned" if back else "grid still running")
        except Exception as error:  # noqa: BLE001 (the close below always runs)
            self.card.log(f"stopping the grid raised {type(error).__name__}: {error}")
        cleaned = self.port.close()
        self.close_failed = "" if cleaned else self.port.close_failed
        self.aborted = self.port.aborted()
        why = f": {self.close_failed}" if self.close_failed else ""
        if self.aborted is not None:
            abandoned = self.aborted["x280_cores"][abi.headers()["KMD_H_ABANDONED"]]
            why += f"; {abandoned} commands abandoned"
        self.card.log(f"x280 runtime closed after a stop (cleaned {cleaned}){why}")
        self.log_state()
        return cleaned

    def log_state(self):
        """Logs the first STATE_DUMP_LINES lines of the kernel's state as DRAM holds
        them (the kernel flushes them when it finishes); never raises."""
        try:
            words = self.kernel_state()
            if words is None:
                self.card.log("kernel state not found in DRAM (lines not flushed)")
                return
            for line_index, line in enumerate(words.reshape(-1, 8)):
                text = " ".join(f"{int(word):#x}" for word in line)
                self.card.log(f"kernel state +{64 * line_index:#05x}: {text}")
        except Exception as error:  # noqa: BLE001 (a failure path: reported)
            self.card.log(f"kernel state not read ({type(error).__name__}: {error})")

    def kernel_state(self):
        """The first STATE_DUMP_LINES lines of the kernel's state (uint64 words) from
        the first page of each bank: the one whose args and lim words name the LIM
        base is the bank the cluster's Memory Port reaches. None if no page does."""
        module = self.card.module
        banks = module.dram_banks(self.device)
        (pages,) = tensor_bytes.read_tensors(
            self.device, module, [self.state_tensor], 0, banks
        )
        words = np.asarray(pages, np.uint8)[:, : 64 * STATE_DUMP_LINES]
        words = np.ascontiguousarray(words).view("<u8")
        base = self.port.lim_base
        found = [
            line for line in words if line[2] == base and line[3] == base
        ]  # args, lim
        return found[0] if len(found) == 1 else None

    def kernel_report(self):
        """While the kernel runs: whether core 0 still runs its loop (its pass marker
        over PROBE_GAP_S), its state and where, and the runtime's view of each core."""
        probe = self.port.probe(PROBE_GAP_S)
        if not probe["running"]:
            return "no dispatcher is running"
        if probe["failed"]:
            states = probe["x280_states"]
            return f"the dispatcher's {probe['failed']}; X280 cores: {states}"
        result = device_state.result_block(probe["words"])
        before, after = probe["marker_before"], probe["marker_after"]
        doing = "is running its loop" if after != before else "is not running its loop"
        report = (
            f"the dispatcher's core 0 {doing} (pass marker {before} -> {after} "
            f"in {PROBE_GAP_S} s; state {result['state_name']}, "
            f"in {result['where_name']}); "
            f"X280 cores: {probe['x280_states']}"
        )
        self.card.log(report)
        return report

    def stop_grid(self):
        """Writes the stop flag into every core (of this island on a split card)
        through the device API."""
        abi_headers = abi.headers()
        module, grid_x = self.card.module, self.port.grid_x
        for core in self._cores():
            address = self.control_addr + abi_headers["KM_CONTROL_STOP"]
            module.write_l1(self.device, core % grid_x, core // grid_x, address, [1])
        self.card.log("stop flag written by the host")

    def _cores(self):
        """The cores of this backend's island on a split card; else every core."""
        grid_x, grid_y = self.port.grid_x, self.port.grid_y
        if self.rows is None:
            return range(grid_x * grid_y)
        return range(self.rows[0] * grid_x, sum(self.rows) * grid_x)

    def _wait_grid(self, timeout):
        """Whether this backend's cores (_cores) returned within `timeout`."""
        first, rows = self.rows or (0, self.port.grid_y)
        return self.card.module.wait_grid(
            self.device, self.port.grid_x, rows, timeout, first
        )

    def read_back(self, stop_grid=False):
        """After the X280 runtime is closed: waits (with a timeout) for the grid's
        kernels, then reads the control blocks ([cores, KM_CONTROL_BYTES / 4] uint32)
        and the trace. `stop_grid`: the dispatcher was not there to stop the grid, so
        the host first writes the stop flag into every core through the device API."""
        if self.port.is_open:
            raise RuntimeError("close the X280 runtime before reading back")
        abi_headers = abi.headers()
        module, grid_x, grid_y = (
            self.card.module,
            self.port.grid_x,
            self.port.grid_y,
        )
        if stop_grid:
            self.stop_grid()
        started = time.monotonic()
        timeout = self.card.grid_timeout
        if not self._wait_grid(timeout):
            self.card.mark_stuck()
            raise GridTimeout(
                f"the grid's kernels did not return within {timeout} s after the stop. "
                "The session does not close the device; at the process's exit the "
                f"runtime waits {process.exit_wait()} for the cores, then puts them in "
                "reset. "
                "A thread that is only stuck is recovered by the next open; a kernel "
                "that left a compute unit running is not, and if the next open fails "
                "the card needs a reset"
            )
        words = abi_headers["KM_CONTROL_BYTES"] // 4
        if self.split is not None:  # the other islands may run: no buffer read
            blocks = np.zeros((grid_x * grid_y, words), np.uint32)
            for core in self._cores():
                blocks[core] = module.read_l1(
                    self.device, core % grid_x, core // grid_x, self.control_addr, words
                )
            returned = self.split[4]
            returned.add(self.rows)
            if len(returned) == len(self.split[3]):
                self.device.finish()  # returns at once: every island is done
            self.card.log(f"island cores returned ({time.monotonic() - started:.1f} s)")
        else:
            self.device.finish()  # returns at once: every core is done
            self.card.log(f"grid returned ({time.monotonic() - started:.1f} s)")
            rows = tensor_bytes._read_uint32(self.control)
            start = self.control_offset
            blocks = rows[:, start : start + words].astype(np.uint32)
        trace = None
        if self.trace_tensor is not None:
            words = tensor_bytes._read_uint32(
                self.trace_tensor, self.device, module
            ).astype(np.uint32)
            trace = words.reshape(
                words.shape[0], self.trace_cap, abi_headers["kTraceWords"]
            )
        self.card.log("state and trace read back")
        return blocks, trace
