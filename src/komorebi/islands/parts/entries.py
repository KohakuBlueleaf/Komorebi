"""The island session's tensor entries: tensors placed by island, bank ranges."""

import warnings
from typing import Any

from komorebi import formats
from komorebi.device import tensor as tensor_module
from komorebi.errors import (
    KomorebiError,
)
from komorebi.islands import graph as islands
from komorebi.islands import records as island_records


class EntriesMixin:
    """The island session's tensor entries: tensors placed by island, bank ranges."""

    # the state of the session's other parts this one reads or sets, owned there
    card_dram: bool  # MeshMixin
    graph: Any  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin

    def _init_entries(self, tensors):
        """An empty tensor table of `tensors` entries."""
        self._entries = [None] * tensors  # the tensor table, by word (`tensors`)

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
            return session.empty(mine, dtype, layout, own=own, **paged)
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
        free = [i for i, held in enumerate(self._entries) if held is None]
        if not free:
            raise KomorebiError(
                "the tensor table has no free entry for a bank range (free a tensor "
                "first, or make it before the load)"
            )
        held = island_records.BankRange(
            island,
            bank,
            address,
            size,
            free[0],
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
    ):
        """A tensor placed by the hint, not written. `split` d over `group` (first
        rank, ranks; default every rank): the group's ranks partition dimension d as
        km::partition does, each rank's part held by its island; `like`: as `like`
        is placed; `copy`: on every island. No hint: split 0. `page_tiles`, `own`:
        Session's."""
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
            part = self.sessions[0].empty(shape, dtype, layout, own=own, **paged)
            tensor = island_records.IslandTensor(
                [part] * self.split, placement, shape, dtype, layout, part.address
            )
            tensor.session = self
            return tensor
        word = None
        if len(self.sessions) > 1:
            free = [i for i, held in enumerate(self._entries) if held is None]
            if not free:
                raise KomorebiError(
                    f"{len(self._entries)} tensors at once: the tensor table is full "
                    "(free one first, or open with a larger tensors=)"
                )
            word = free[0]
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
                        own=own,
                        **paged,
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
        tensor.session = self
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
    ):
        """A DRAM tensor holding `data` as Session.tensor takes it, placed by the hint
        (empty() names them); each island's part written through its own host
        channel."""
        raw, shape = formats.to_bytes(data, dtype, layout)
        tensor = self.empty(shape, dtype, layout, split, group, like, copy, page_tiles)
        try:
            self.write(tensor, raw.reshape(tensor.pages, tensor.page_bytes))
        except BaseException:
            tensor.free()
            raise
        return tensor
