"""The island session's tensor table: its rows, the cores' cache, its copies."""

from typing import Any

import numpy as np

from komorebi import abi
from komorebi.errors import (
    KomorebiError,
)
from komorebi.islands import graph as islands
from komorebi.islands.graph import TENSORS_MAX
from komorebi.session.parts import memory


class TableMixin:
    """The island session's tensor table: its rows, the cores' cache, its copies."""

    # its state the constructor sets (komorebi.islands.session)
    tensor_cache: int | bool | None
    # the state of the session's other parts this one reads or sets, owned there
    _regions: list  # MeshMixin
    links: int  # MeshMixin
    run: Any  # MeshMixin
    sessions: tuple  # MeshMixin
    split: int  # MeshMixin
    _entries: list  # EntriesMixin

    def _init_table(self):
        """No load's table yet."""
        self._tables = []  # per island: its cores' tensor table (L1), from a load
        self._drams = []  # per island: the table in DRAM, when its cores cache rows
        self._table_size = 0  # its entries

    def _cache_bytes(self):
        """A core's L1 for a cached table (km_table_cache.hpp): the head, then the
        reader's and the writer's caches (tags, last uses, clock, misses; the rows)."""
        k, row = self._cache_k(), self._cache_row()
        head = -(-((2 * k + 2) * 4) // 64) * 64
        return abi.headers()["KM_TABLE_HEAD_BYTES"] + 2 * (head + k * row)

    def _cache_k(self):
        """The rows a RISC caches: tensor_cache's K, else KM_TABLE_CACHE_ENTRIES."""
        given = self.tensor_cache
        return given if given else abi.headers()["KM_TABLE_CACHE_ENTRIES"]

    def _cache_row(self):
        return -(-islands.tensor_entry_bytes(len(self.sessions)) // 64) * 64

    def _cached(self):
        """Whether the table lives in each island's DRAM, its cores caching rows: as
        tensor_cache says, else where that takes less L1 than the table flat."""
        if self.tensor_cache is not None:
            return bool(self.tensor_cache)
        flat = len(self._entries) * islands.tensor_entry_bytes(len(self.sessions))
        return self._cache_bytes() < flat

    def _dram_rows(self):
        """The cached table's rows in DRAM: an entry a page of whole 64 B lines."""
        size = islands.tensor_entry_bytes(len(self.sessions))
        rows = np.zeros((self._table_size, self._cache_row()), np.uint8)
        rows[:, :size] = self._table_rows().view(np.uint8).reshape(-1, size)
        return rows

    def _held_l1(self, island):
        """What island `island`'s session keeps in the L1 of every core, at the top
        (the load's L1 budget counts it): [(row name, bytes)] of the tensors it holds
        now: its own words, its link area when the link cores do not hold
        them alone, its tensor table, its message service's region."""
        held = dict(getattr(self, "_held", [{}] * (island + 1))[island])
        form = (
            f"tensor table, cached ({self._cache_k()} rows a RISC)"
            if self._drams
            else f"tensor table, flat ({self._table_size} entries)"
        )
        held[form] = self._tables[island : island + 1]
        held["message region"] = self._regions[island : island + 1]
        rows = []
        for name, tensors in held.items():
            size = sum(t.aligned_page_bytes for t in tensors if not t.freed)
            if size:
                rows.append((f"session: {name}", size))
        return rows if getattr(self, "_held", None) else rows + self._first_held()

    def _split_held(self):
        """A split card's one session: before the first load the rows it makes
        (_first_held, its link area too); after it, l1_taken counts them made."""
        return [] if getattr(self, "_held", None) else self._first_held()

    def _first_held(self):
        """Before the first load: the rows it makes and the session keeps after it
        (own words, island table, tensor table), so a class made now gets the room a
        later one does; a link area (link cores not apart) is known at the load."""
        if len(self.sessions) < 2:
            return []
        h, count, entries = (
            abi.headers(),
            len(self.graph.islands),
            self._table_entries(),
        )
        rows = [
            ("session: own words", -(-h["KM_OWN_ALL_BYTES"] // 16) * 16),
            ("session: island table", -(-6 * count // 16) * 16 * 4),
        ]
        if not self._tables:
            if self._cached():
                name = f"cached ({self._cache_k()} rows a RISC)"
                size = self._cache_bytes()
            else:
                name = f"flat ({entries} entries)"
                size = entries * islands.tensor_entry_bytes(count)
            table = "split card's tensor table" if self.split > 1 else "tensor table"
            rows.append((f"session: {table}, {name}", size))  # split: a bounds list too
        if self.split > 1:  # one card: the link area a session tensor on every core
            rows.append(("session: link area", h["KM_LINK_AREA_BYTES"]))
        return [(name + memory.AT_LOAD, size) for name, size in rows]

    def _l1_row(self, island):
        """What every core of `island` holds at the table's place: the table, or a
        cached table's head (word 0: its DRAM address) and empty caches."""
        if not self._drams:
            return self._table_rows()
        row = np.zeros(self._cache_bytes() // 4, np.uint32)
        row[0] = self._drams[island].address
        return row

    def _table_bound(self):
        """KM_TENSORS_MAX for a table of more than TENSORS_MAX entries; KM_TABLE_CACHE
        when the table is cached (_cached); a flat one of 32 or fewer: neither."""
        entries = len(self._entries)
        out = {"KM_TENSORS_MAX": entries} if entries > TENSORS_MAX else {}
        if self._cached() and len(self.sessions) != 1:  # one island: no table
            out["KM_TABLE_CACHE"] = self._cache_k()
        return out

    def _table_entries(self, extra=0):
        """The entries of a load's tensor table: every word (`tensors`), so a tensor
        made while the run lives has its row; `extra` is kept for callers."""
        del extra
        return len(self._entries)

    def _table_row(self, word):
        """Entry `word` into every island's table: by the run's cores while a run
        lives (_table_row_live), else by the host (a load writes it otherwise)."""
        if self._tables and self.run is not None and not self.run.stopped:
            self._table_row_live(word)
        else:
            self._write_table()

    def _table_row_live(self, word, timeout=10.0):
        """Entry `word` written by every island's ranks (ENTRY with KM_ENTRY_TABLE): the
        row in a page of each island's DRAM at its L1 place's offset mod 64, the page
        whole 64 B lines that hold it (an entry grows with the islands), one command
        an island, every island's queue stored (run.send) before any is waited for."""
        h = abi.headers()
        size = islands.tensor_entry_bytes(len(self.sessions))
        row = np.frombuffer(self._entries[word].entry(self.dram_banks), np.uint8)
        scratches, handles = [], []
        try:
            for island, run in enumerate(self.run.runs):
                if self._drams:  # the DRAM row written; the ranks forget their copies
                    page = np.zeros((1, self._cache_row()), np.uint8)
                    page[0, :size] = row
                    self.sessions[island].write(self._drams[island], page, first=word)
                    address = 0
                else:
                    offset = (self._tables[island].address + word * size) % 64
                    page = np.zeros(-(-(offset + size) // 64) * 64, np.uint8)
                    page[offset : offset + size] = row
                    scratch = self.sessions[island].tensor(
                        page.view(np.uint32)[None], "uint32"
                    )
                    scratches.append(scratch)
                    address = scratch.address + offset
                part = run._flag_part if run.island_ranks != run.cores else 0
                words = (h["KM_ENTRY_TABLE"], word, address)
                group = (run.island_first, run.island_ranks)
                handles.append(
                    run._queue_command(
                        h["KM_HANDLER_ENTRY"], group, words, False, part, budget=timeout
                    )
                )
            # every island's queue stored: a crossing command queued before the row
            # starts on every island, not on the one waited first alone
            self.run.send()
            for island, (run, handle) in enumerate(zip(self.run.runs, handles)):
                event = run.wait(handle, timeout)
                if event.kind != "done":
                    raise KomorebiError(
                        f"island {island}: the table row of tensor word {word} ended "
                        f"{event.kind}: {event}"
                    )
        finally:
            for scratch in scratches:
                scratch.free()

    def _table_rows(self):
        """The tensor table every core of an island holds: an entry a word, in word
        order ([entries x entry bytes / 4] uint32, the table's entries)."""
        size = islands.tensor_entry_bytes(len(self.sessions))
        table = bytearray(self._table_size * size)
        for word, tensor in enumerate(self._entries[: self._table_size]):
            if tensor is not None:
                table[word * size : (word + 1) * size] = tensor.entry(self.dram_banks)
        return np.frombuffer(bytes(table), np.uint32)

    def _tables_for_load(self):
        """Each island's tensor table in the L1 of all its cores, made at the first
        load and written: its address for Session.load (several islands; else 0)."""
        if len(self.sessions) == 1:
            return [0]
        h = abi.headers()
        for name, value in (
            ("KM_TENSORS_MAX", islands.TENSORS_MAX),
            ("KM_CORES_MAX", islands.CORES_MAX),
            ("KM_ISLAND_PORTS_MAX", islands.PORTS_MAX),
            ("KM_GROUP_LINK_BASE", islands.LINK_BASE),
        ):  # the host's copies of the header's
            if h[name] != value:
                raise KomorebiError(
                    f"the host's {value} differs from km_abi.h's {name}"
                )
        if self._tables and self._table_size < self._table_entries():
            for table in {id(t): t for t in [*self._tables, *self._drams]}.values():
                table.free()  # the load's tensors need more entries: made again
            self._tables, self._drams = [], []
        if not self._tables:
            self._table_size = self._table_entries()
            own = self.sessions[: 1 if self.split > 1 else None]
            if self._cached():
                rows = self._dram_rows().view(np.uint32)
                self._drams = [session.tensor(rows, "uint32") for session in own]
                self._drams *= len(self.sessions) // len(self._drams)
            for island, session in enumerate(own):
                columns, rows = session.grid
                self._tables.append(
                    session.tensor(
                        np.tile(self._l1_row(island), (columns * rows, 1)),
                        "uint32",
                        shard_grid=(columns, rows),
                    )
                )
            self._tables *= len(self.sessions) // len(self._tables)  # split: one
        else:
            self._write_table()
        return [table.address for table in self._tables]

    def _write_table(self):
        """Writes the tensor table into every core of every island, when a run holds
        it (a load writes it otherwise); a cached one into DRAM, its caches emptied."""
        if not self._tables:
            return
        pairs = {id(table): i for i, table in enumerate(self._tables)}
        for island in pairs.values():  # a split card: one table
            session, table = self.sessions[island], self._tables[island]
            if self._drams:
                session.write(self._drams[island], self._dram_rows())
            row = self._l1_row(island)
            session.write(table, np.tile(row, (table.pages, 1)).view(np.uint8))

    def held_bytes(self, apart=True, extra=0):
        """The L1 bytes a load keeps on every core (the rows of l1_held it will have):
        the own words; the link area unless `apart` (the link cores run
        their own kernels and hold them alone, _apart); the tensor table with `extra`
        tensors more than now."""
        if len(self.sessions) == 1:
            return 0
        h = abi.headers()
        size = self._table_entries(extra) * islands.tensor_entry_bytes(
            len(self.sessions)
        )
        size = self._cache_bytes() if self._cached() else size
        if self.links:
            size += h["KM_OWN_ALL_BYTES"]
            if not apart:
                size += h["KM_LINK_AREA_BYTES"]
        return size

    def table_misses(self):
        """A cached table's misses since the load, per island: [cores, 2] (each core's
        reader, writer), read from the cores' L1 after run.stop(); None if flat."""
        if not self._drams:
            return None
        if self.run is not None and not self.run.stopped:
            raise KomorebiError(
                "table_misses: after run.stop() (a live run's L1 is read through its "
                "own transfer handler, which a hand program lacks)"
            )
        k = self._cache_k()
        head = -(-((2 * k + 2) * 4) // 64) * 64
        part = (head + k * self._cache_row()) // 4
        first = abi.headers()["KM_TABLE_HEAD_BYTES"] // 4 + 2 * k + 1
        out = []
        for session, table in zip(self.sessions, self._tables):
            (raw,) = session.read(table)
            words = np.ascontiguousarray(raw).view("<u4")
            out.append(words[:, [first, first + part]].astype(np.int64))
        return out
