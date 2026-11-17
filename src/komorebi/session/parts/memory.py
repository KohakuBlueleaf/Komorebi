"""The session's device memory: tensors, the L1 a load leaves, page bounds, NoCs."""

import os
from typing import Any

from komorebi import abi, formats, layout
from komorebi.device import extents, kernel_buffer, native
from komorebi.device import tensor as tensor_module
from komorebi.dsl.program import session_nocs
from komorebi.program import several as programs
from komorebi.program.description import (
    Program,
)
from komorebi.session import kernels as kernel_module
from komorebi.session import sizes, tensor_bytes

TAKEN_ROW = "L1 held by the session (its L1 tensors, runs not stopped)"
AT_LOAD = " (made at the first load)"  # an l1_held row not made yet: not in the span


class MemoryMixin:
    """The session's device memory: tensors, the L1 a load leaves, page bounds, NoCs."""

    # the state of the session's other parts this one reads or sets, owned there
    device: Any  # Session
    tree: Any  # Session
    _module: Any  # OpenMixin
    channel: Any  # OpenMixin
    island: Any  # LoadMixin
    run: Any  # LoadMixin

    def _init_memory(self):
        """No tensor made yet; the owner holds no L1."""
        self._kept = kernel_buffer.Kept()  # the tensors made before the first load
        self._dram = extents.Registry()  # its DRAM tensors: KM_GUARD's page bounds
        self.page_bounds = extents.MOST  # the most a load's list holds (0: none)
        self.transfer_chunk = None  # a larger transfer chunk asked for, where it fits
        # the owner's L1 on every core of this session's card, kept at the top (an
        # island session's link areas, tensor table): () -> [(name, bytes)]
        self.l1_held = lambda: []
        self._taken = None  # l1_taken() as a load found it before its own L1
        self.largest_page = 0  # bytes of the largest page of a DRAM tensor made so far

    @property
    def grid(self):
        """The Tensix grid (x, y) of the open device."""
        return tuple(self.device.grid())

    @property
    def dram_banks(self):
        """The DRAM banks of the open device (a page p of a DRAM tensor: bank p %
        banks)."""
        self._module = self._module or native.load_module(self.tree)
        return int(self._module.dram_banks(self.device))

    def tensor(
        self,
        data,
        dtype,
        layout="row",
        shard_grid=None,
        transposed_tiles=False,
        page_tiles=1,
    ):
        """A tensor on the card holding `data` (anything np.asarray takes) as `dtype`
        (formats.DTYPES; a block format in tiles) in `layout` ("row" or "tile"), each
        tile transposed if `transposed_tiles`, `page_tiles` tiles a DRAM page; in DRAM,
        or a row in each core's L1 of `shard_grid`. Written before or while it runs."""
        self._module = self._module or native.load_module(self.tree)
        raw, shape = formats.to_bytes(data, dtype, layout, transposed_tiles)
        tensor = tensor_module.empty(
            self._module,
            self.device,
            shape,
            dtype,
            layout,
            shard_grid,
            page_tiles=page_tiles,
        )
        tensor.transposed_tiles = transposed_tiles
        self._note_page(tensor)
        if self._transfer_way([tensor], 0, None) is None:
            tensor.buffer.write(raw)
        else:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        tensor.written = True
        self._kept.keep(tensor, data=(data, transposed_tiles))  # after: not the raw
        self._dram.add(tensor)
        tensor.loaded = self._run_loaded
        return tensor

    def empty(
        self,
        shape,
        dtype,
        layout="row",
        shard_grid=None,
        address=None,
        own=False,
        page_tiles=1,
    ):
        """A tensor as tensor() makes it, not written. `own`: a region the runtime
        makes for itself, written by bank address: no transfer chunk follows its page.
        """
        self._module = self._module or native.load_module(self.tree)
        tensor = tensor_module.empty(
            self._module,
            self.device,
            shape,
            dtype,
            layout,
            shard_grid,
            address,
            page_tiles,
        )
        self._kept.keep(tensor, address)
        self._dram.add(tensor)
        if not own:
            self._note_page(tensor)
        tensor.loaded = self._run_loaded
        return tensor

    def _run_loaded(self):
        """Whether a run of this session is loaded and not stopped: Tensor.read()
        refuses then (tensor.refuse_live)."""
        return self.run is not None and not getattr(self.run, "stopped", False)

    def _note_page(self, tensor):
        """Keeps the largest page of a DRAM tensor: a load sizes the run's transfer
        buffer by it."""
        if tensor.in_dram:
            self.largest_page = max(self.largest_page, tensor.aligned_page_bytes)

    def launch(self):
        """An empty native Launch on the open device, for a host-side runner that
        builds and launches a program of its own (a test feeder) instead of load()."""
        self._module = self._module or native.load_module(self.tree)
        return self._module.Launch(self.device)

    def l1(self):
        """(kernel buffer, L1 left for buffers) of each core in bytes, from the
        runtime's layout of the open device: what a larger buffer costs the buffers."""
        self._module = self._module or native.load_module(self.tree)
        kernel, free, _ = self._module.kernel_buffer(self.device)
        return int(kernel), int(free)

    def l1_rows(self, program, trace=0, placed=None, buffers=None):
        """A core's L1 budget rows for `program` loaded with `trace`, the slots aside:
        its buffers (`buffers` bytes if given) with the token and trace pages, the
        control block and overlay area, the host transfer buffer, what the session
        keeps. The load's table is these and the slots; a class's build checks them."""
        abi_headers = abi.headers()
        trace_cap = (trace + 3) // 4 * 4 if trace else 0
        if buffers is None:
            buffers = sum(b.page_bytes * b.pages for b in program.buffers)
        buffers += 2 * abi_headers["kTokenBytes"] + trace_cap * 16
        control = tensor_bytes._control_bytes(kernel_module.slots_most(program, placed))
        rows = [
            ("circular buffers", buffers),
            ("control block and overlay area", control),
        ]
        # A mock device maps no host channel: its budget holds the card's buffer.
        mock = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))
        if self.channel is not None or program.threads or mock:
            held = placed.programs if placed else [program]
            switch = bool(placed and placed.switch)
            chunk = tensor_bytes._transfer_chunk_of(
                self.largest_page, self.l1, self.transfer_chunk, held, switch,
                trace_cap, control,
            )  # fmt: skip
            size = 2 * chunk + abi_headers["KM_TRANSFER_WORD_BYTES"] + sizes.NOC_ALIGN
            rows.append(("host transfer buffer", size))
        messages = getattr(program, "messages", None)
        if messages is not None and messages.direct:
            cores = self.grid[0] * self.grid[1]
            rows.append(
                ("direct messages (tables, slots)", messages.direct_bytes(cores))
            )
        bounds = self.page_bounds_list(placed.programs if placed else [program])
        if bounds is not None:
            rows.append(("page bounds list", bounds[0].nbytes))
        return rows + list(self.l1_held()) + self.l1_taken()

    def page_bounds_list(self, programs):
        """KM_GUARD's list of this session's DRAM tensors for a load of `programs`
        (komorebi.device.extents.listed); None: none (every program unguarded, or the
        load's tensor table bounds them, or page_bounds 0)."""
        if not self.page_bounds or not any(getattr(p, "guard", True) for p in programs):
            return None
        if any(name.startswith("session: tensor table") for name, _ in self.l1_held()):
            return None
        return extents.listed(self._dram.live(), self.page_bounds)

    def _hand_nocs(self, program):
        """A program written as a class whose rectangle needs the grid's NoCs gets
        this session's (komorebi.dsl.program.session_nocs)."""
        if isinstance(program, programs.Plan):
            loaded = program.programs
        else:
            loaded = list(program) if isinstance(program, dict | list) else [program]
        loaded = [  # and a merged program's sources made apart (komorebi.merge)
            each
            for one in loaded
            for each in [one, *getattr(one, "merged_sources", ())]
            if isinstance(each, Program)
        ]
        needing = [one for one in loaded if getattr(one, "rectangle", None)]
        if needing:
            nocs = self.nocs()
            for one in needing:
                session_nocs(one, nocs, self.island)

    def nocs(self):
        """The grid's NoC coordinates by core (a layout.Nocs), from the open device:
        for program arguments that name other cores, before load()."""
        self._module = self._module or native.load_module(self.tree)
        grid_x, grid_y, table, noc0 = self._module.core_nocs(self.device)
        return layout.nocs((grid_x, grid_y), table, noc0)

    def l1_taken(self):
        """The L1 budget table's row of the L1 this session already holds on every
        core (its L1 tensors, a run not stopped): the L1 top down to the lowest L1
        buffer, less what the owner keeps (l1_held). [] when none."""
        if self._taken is not None:
            return self._taken
        if self.device is None:
            return []
        self._module = self._module or native.load_module(self.tree)
        lowest = int(self._module.lowest_l1(self.device))
        if not lowest:
            return []
        top = int(self._module.kernel_buffer(self.device)[2])
        made = [size for name, size in self.l1_held() if not name.endswith(AT_LOAD)]
        held = top - lowest - sum(made)
        return [(TAKEN_ROW, held)] if held > 0 else []
