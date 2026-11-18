"""The island session's tensor entries: tensors placed by island, bank ranges."""

import warnings
from typing import Any

import numpy as np

from komorebi import formats
from komorebi.device import tensor as tensor_module
from komorebi.errors import (
    KomorebiError,
)
from komorebi.islands import graph as islands
from komorebi.islands import records as island_records

L1_DIFFERS = (
    "the islands' L1 differs: island {island} would place the L1 tensor at {at:#x}, "
    "island 0 at {first:#x} (an L1 tensor made on one island alone, with "
    "session.island(i), holds L1 the others do not): make every L1 tensor on the "
    "session (shard_grid=), which places it on every island at one address"
)


def _own(own):
    """Session.empty's `own` as a keyword, given only when set (as `page_tiles`)."""
    return {"own": True} if own else {}


class EntriesMixin:
    """The island session's tensor entries: tensors placed by island, bank ranges."""

    # the state of the session's other parts this one reads or sets, owned there
    card_dram: bool  # MeshMixin
    graph: Any  # MeshMixin
    run: Any  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin
    _tables: list  # TableMixin
    _table_size: int  # TableMixin

    def _init_entries(self, tensors):
        """An empty tensor table of `tensors` entries."""
        self._entries = [None] * tensors  # the tensor table, by word (`tensors`)

    def _free_word(self, what):
        """A free word of the tensor table for `what`. A full table doubles (to
        TENSORS_MOST), and the next load makes it again; a live run's table does not:
        its kernels read a word past their bound as a bank address."""
        live = self._tables and self.run is not None and not self.run.stopped
        held = self._table_size if live else len(self._entries)
        free = [i for i, entry in enumerate(self._entries[:held]) if entry is None]
        if free:
            return free[0]
        if live:
            raise KomorebiError(
                f"{what}: the live run's tensor table is full ({held} entries; one "
                "card has no table): free a tensor first, or make it after "
                "run.stop(): the next load takes a larger table"
            )
        if held >= islands.TENSORS_MOST:
            raise KomorebiError(
                f"{what}: {held} tensors at once, the tensor table's most "
                "(free one first)"
            )
        self._entries += [None] * min(held, islands.TENSORS_MOST - held)
        return held

    @staticmethod
    def _fits(session, shape, pages, dtype, layout, page_tiles=1):
        """The most pages of this kind an island's DRAM takes now (found by trying)."""
        low, high = 0, pages
        while low < high:
            middle = (low + high + 1) // 2
            trial = (
                (middle * formats.TILE, page_tiles * formats.TILE)
                if layout == "tile"
                else (middle, shape[-1])
            )
            paged = {"page_tiles": page_tiles} if page_tiles != 1 else {}
            try:
                session.empty(trial, dtype, layout, **paged).free()
                low = middle
            except RuntimeError:
                high = middle - 1
        return low

    def _run_loaded(self):
        """Whether a run of this session is loaded and not stopped: an island
        tensor's read() and write() refuse then (device.tensor.refuse_live)."""
        return self.run is not None and not getattr(self.run, "stopped", False)

    def _part(
        self, session, island, placement, shape, dtype, layout, page_tiles=1, own=False
    ):
        """Island `island`'s buffer of its part (None: it holds no page): the whole
        shape when it holds every page, else its pages with the same page bytes
        (`own`: Session.empty's). A part that does not fit raises with the most pages
        that would."""
        pages = placement.part_pages(island)
        if pages == 0:
            return None
        whole = pages == placement.pages
        if whole:
            mine = shape
        elif layout == "tile":
            mine = (pages * formats.TILE, page_tiles * formats.TILE)
        else:
            mine = (pages, shape[-1])
        paged = {"page_tiles": page_tiles} if page_tiles != 1 else {}
        try:
            return session.empty(mine, dtype, layout, **paged, **_own(own))
        except tensor_module.DramFull as error:  # refused before the allocator
            fits, said = error.fits, error.said
        except KomorebiError:
            raise
        except RuntimeError as error:  # the allocator's: the DRAM is full
            fits = self._fits(session, mine, pages, dtype, layout, **paged)
            lines = str(error).splitlines() or [""]  # its sentence, not its backtrace
            said = next((line for line in lines if "Out of Memory" in line), lines[0])
        raise KomorebiError(
            f"island {island}: its part of {pages} pages of shape {tuple(shape)} "
            f"does not fit its DRAM ({said}); at most {fits} pages of this kind "
            "fit there now: split it over more islands' ranks (`group`) or free "
            "tensors"
        ) from None

    def _l1(self, shape, dtype, layout, shard_grid, address, own):
        """An L1 tensor, one logical [cards x cores, words] tensor: card i's rows in the
        L1 of the cores of `shard_grid` (one island's grid) on its island, at one
        address on every island (a split card: the card's [cores, words])."""
        grid = tuple(int(extent) for extent in shard_grid)
        cores = grid[0] * grid[1]
        cards = 1 if self.split > 1 else len(self.sessions)
        padded = formats.padded_shape(shape, layout)
        if layout != "row" or len(padded) != 2 or padded[0] != cards * cores:
            whose = "the card's" if self.split > 1 else f"{cards} islands x"
            raise ValueError(
                f"an L1 tensor on islands is [{whose} {cores} cores, words] rows, a "
                f"row a core of shard_grid {grid}, island by island; got "
                f"{tuple(shape)} in {layout!r}"
            )
        parts = []
        try:
            parts.extend(  # in turn: the ones made are freed if one fails
                self.sessions[island].empty(
                    (cores, padded[1]),
                    dtype,
                    layout,
                    shard_grid=grid,
                    address=address,
                    **_own(own),
                )
                for island in range(cards)
            )
            for island, part in enumerate(parts):
                if part.address != parts[0].address:
                    raise KomorebiError(
                        L1_DIFFERS.format(
                            island=island, at=part.address, first=parts[0].address
                        )
                    )
        except BaseException:
            for part in parts:
                part.free()
            raise
        if cards == 1:  # every island's rows are the card's
            placement = islands.Placement(
                ((0, 1),) * len(self.sessions), copy=True, inner=cores, split=None
            )
            parts *= len(self.sessions)
        else:
            placement = islands.Placement(
                tuple((island, 1) for island in range(cards)), steps=cards, inner=cores
            )
        tensor = island_records.IslandTensor(
            parts, placement, padded, dtype, layout, parts[0].address
        )
        tensor.shard_grid, tensor.in_dram = grid, False
        tensor.session, tensor.loaded = self, self._run_loaded
        return tensor

    def _release_entry(self, tensor):
        """A freed tensor's entry of the tensor table is free again."""
        if (
            tensor.address < len(self._entries)
            and self._entries[tensor.address] is tensor
        ):
            self._entries[tensor.address] = None

    def bank_range(self, island, bank, address, size):
        """A range of one DRAM bank as kernels on any island name it (km::BankRange):
        `size` bytes of bank `bank` of island `island` from bank address `address`.
        Takes an entry of the tensor table as a tensor does; free() frees it."""
        if not 0 <= island < len(self.sessions):
            raise ValueError(f"island {island} of {len(self.sessions)}")
        if not 0 <= bank < islands.DRAM_BANKS:
            raise ValueError(
                f"bank {bank}: the banks are 0 to {islands.DRAM_BANKS - 1}"
            )
        if (
            address % 64
            or address < len(self._entries)
            or size <= 0
            or address + size > 1 << 32
        ):
            raise ValueError(
                f"{size} bytes at bank address {address:#x}: whole 64-byte lines from "
                f"{len(self._entries)} on, inside the bank's 4 GiB"
            )
        if len(self.sessions) == 1 or self.card_dram:  # every page here: no entry
            return island_records.BankRange(island, bank, address, size, address)
        held = island_records.BankRange(
            island,
            bank,
            address,
            size,
            self._free_word("a bank range"),
            len(self.sessions),
            self._release_entry,
        )
        self._entries[held.address] = held
        try:
            self._table_row(held.address)
        except BaseException:
            held.free()
            raise
        return held

    def empty(
        self,
        shape,
        dtype,
        layout="row",
        split=None,
        group=None,
        like=None,
        copy=False,
        page_tiles=1,
        own=False,
        shard_grid=None,
        address=None,
    ):
        """A tensor placed by the hint, not written: `split` d over `group` (first
        rank, ranks; default all), its ranks partitioning d as km::partition, each part
        on its rank's island; `like`: as `like`; `copy`: on every island; none: split 0.
        `page_tiles`, `own`: Session's; `shard_grid`: in L1 on every island (_l1)."""
        if shard_grid is not None:
            hints = (split, group, like, copy or None, page_tiles != 1 or None)
            if any(hint is not None for hint in hints):
                raise ValueError(
                    "shard_grid= places the tensor in L1, a row a core of every "
                    "island: no split, group, like, copy or page_tiles with it"
                )
            return self._l1(shape, dtype, layout, shard_grid, address, own)
        if address is not None:
            raise ValueError("address= is an L1 tensor's: give shard_grid= with it")
        # padded to whole tiles as a one-card tensor is, its shape before kept
        logical = formats.logical_shape(shape, layout)
        shape = formats.padded_shape(shape, layout)
        if like is not None:
            if split is not None or group is not None or copy:
                raise ValueError("`like` takes the other tensor's hint: no other hint")
            hint = like.placement
            split, group, copy = hint.split, hint.group, hint.copy
        if page_tiles != 1:
            formats.page_bytes(
                formats.padded_shape(shape, layout), dtype, layout, page_tiles
            )  # its refusals before the placement's
        placement = islands.place(
            self.graph, shape, layout, split, group, copy, page_tiles
        )
        uneven = None if self.card_dram else islands.uneven_split(self.graph, placement)
        if uneven is not None:
            shares, across = uneven
            pages = across * placement.inner * placement.outer
            line = (
                f"tensor {tuple(shape)} split={placement.split} over "
                f"group={placement.group}: the islands' (steps, ranks) {shares} are "
                f"not in proportion to their ranks; a command that splits the group by "
                f"island reads {across} of {placement.steps} steps ({pages} pages) "
                "across the link (a group whose ranks each take the same count of "
                "steps keeps them local)"
            )
            self.log(line)
            warnings.warn(line, stacklevel=2)
        if self.card_dram:  # one tensor in the card's DRAM, every page local
            paged = {"page_tiles": page_tiles} if page_tiles != 1 else {}
            part = self.sessions[0].empty(shape, dtype, layout, **paged, **_own(own))
            tensor = island_records.IslandTensor(
                [part] * self.split, placement, shape, dtype, layout, part.address
            )
            tensor.session, tensor.loaded = self, self._run_loaded
            tensor_module.unwritten(tensor, logical)
            return tensor
        word = None
        if len(self.sessions) > 1:
            word = self._free_word(f"tensor {tuple(shape)}")
        parts = []
        try:
            for island, session in enumerate(self.sessions):
                paged = {"page_tiles": page_tiles} if page_tiles != 1 else {}
                parts.append(
                    self._part(
                        session,
                        island,
                        placement,
                        shape,
                        dtype,
                        layout,
                        **paged,
                        **_own(own),
                    )
                )
        except BaseException:
            for part in parts:
                if part is not None:
                    part.free()
            raise
        tensor = island_records.IslandTensor(
            parts, placement, shape, dtype, layout, word, self._release_entry
        )
        tensor.session, tensor.loaded = self, self._run_loaded
        tensor_module.unwritten(tensor, logical)
        if word is None:
            tensor.address = parts[0].address  # one island: its bank address
        else:
            if any(
                0 < part.address < len(self._entries) for part in tensor.parts if part
            ):
                raise KomorebiError("a bank address below the tensor table's size")
            self._entries[word] = tensor
            try:
                self._table_row(word)
            except BaseException:
                tensor.free()
                raise
        return tensor

    def tensor(
        self,
        data,
        dtype,
        layout="row",
        split=None,
        group=None,
        like=None,
        copy=False,
        page_tiles=1,
        transposed_tiles=False,
        shard_grid=None,
    ):
        """A tensor holding `data` as Session.tensor takes it (each tile transposed if
        `transposed_tiles`), placed by the hint or in L1 (empty() names them); each
        island's part written through its own host channel."""
        raw, shape = formats.to_bytes(data, dtype, layout, transposed_tiles)
        tensor = self.empty(
            shape,
            dtype,
            layout,
            split,
            group,
            like,
            copy,
            page_tiles,
            shard_grid=shard_grid,
        )
        tensor.transposed_tiles = transposed_tiles
        tensor.logical = formats.logical_shape(np.shape(data), layout)  # zero-padded
        try:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        except BaseException:
            tensor.free()
            raise
        return tensor
