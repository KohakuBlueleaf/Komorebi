"""The loader: the placed weights from the converted cache into the islands' DRAM, and
every weight by name (`W = load(session)`, `W[name]` a Handle); a subset onto a few
islands, the read-back check and resume: README.md, "The loader".
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# the folder of the checkpoint (ckpt/), placement.json and the cache (cache/v1/)
LLM = os.path.expanduser(os.environ.get("GALAXY_LLM", "~/llm"))
PLACEMENT = os.path.join(LLM, "placement.json")
CACHE = os.path.join(LLM, "cache", "v1")
# a write's launch costs about 50 ms (card 0: 64 MiB in 0.12 s, 256 MiB in 0.29 s)
CHUNK_BYTES = 256 << 20
# a range under this goes the device API's way (whole tensors only: LAUNCH_MIN_BYTES),
# so a shorter tail is written as the last range of this size (README.md)
RANGE_MIN_BYTES = 80 << 10
THREADS = 8  # islands written at once by default (load_box measurements below)


class Handle:
    """One weight in the session: its arena, the arena's word, its pages."""

    __slots__ = ("name", "arena", "first", "pages", "shape", "dtype", "island", "card")

    def __init__(self, name, arena, first, pages, shape, dtype, island, card):
        self.name, self.arena, self.first, self.pages = name, arena, first, pages
        self.shape, self.dtype, self.island, self.card = shape, dtype, island, card

    @property
    def word(self):
        """The arena's reference word: what a command passes a kernel (its tensor
        table entry over islands, its bank address on one card)."""
        return self.arena.address

    def __repr__(self):
        return (
            f"Handle({self.name!r}, island {self.island}, {self.dtype}, "
            f"shape {self.shape}, pages {self.first}..{self.first + self.pages - 1})"
        )


class Weights(dict):
    """Handles by name; `arenas` {(session island, dtype): tensor}; `report` the
    load's figures."""

    def __init__(self):
        super().__init__()
        self.arenas = {}
        self.report = {}

    def free(self):
        for tensor in self.arenas.values():
            tensor.free()
        self.arenas.clear()
        self.clear()


def _pick(entries, names):
    if names is None:
        return list(entries)
    if callable(names):
        return [e for e in entries if names(e["name"])]
    wanted = set(names)
    out = [e for e in entries if e["name"] in wanted]
    missing = wanted - {e["name"] for e in out}
    if missing:
        raise KeyError(f"not in the placement: {sorted(missing)[:5]}")
    return out


def _islands_of(session):
    return len(getattr(session, "sessions", [session]))


def _arena(session, island, pages, dtype):
    """A tile tensor of `pages` one-tile pages, every page on `island`."""
    shape = (pages * 32, 32)
    if _islands_of(session) == 1:
        return session.empty(shape, dtype, "tile")
    graph = session.graph
    group = (graph.first(island), graph.count(island))
    return session.empty(shape, dtype, "tile", split=0, group=group)


def _part(session, arena, island):
    """(the one-card session, the buffer) that hold `arena`'s pages on `island`."""
    if _islands_of(session) == 1:
        return session, arena
    part = arena.parts[island]
    arena.written, arena.sent = True, None  # as IslandSession.write marks it
    return session.sessions[island], part


class _Source:
    """Reads runs of pages of a cache arena file."""

    def __init__(self, cache, island, dtype, tile):
        self.path = os.path.join(cache, f"i{island:02d}-{dtype}.bin")
        self.fd = os.open(self.path, os.O_RDONLY)
        self.tile = tile

    def pages(self, first, count, out=None):
        size = count * self.tile
        buf = np.empty(size, np.uint8) if out is None else out.reshape(-1)
        got = os.preadv(self.fd, [buf], first * self.tile)
        if got != size:
            raise OSError(f"{self.path}: {got} of {size} B at page {first}")
        return buf.reshape(count, self.tile)

    def close(self):
        os.close(self.fd)


class _Gather:
    """An arena's pages from the cache by its copy runs (cache island, cache first,
    pages, arena first), in arena order."""

    def __init__(self, cache, dtype, tile, runs):
        self.runs, self.tile = sorted(runs, key=lambda r: r[3]), tile
        self.pages = sum(r[2] for r in runs)
        self.sources = {
            src: _Source(cache, src, dtype, tile) for src in {r[0] for r in runs}
        }

    def rows(self, lo, hi):
        out = np.empty((hi - lo, self.tile), np.uint8)
        for src, src_first, pages, dst_first in self.runs:
            a, b = max(lo, dst_first), min(hi, dst_first + pages)
            if a < b:
                rows = out[a - lo : b - lo]
                self.sources[src].pages(src_first + a - dst_first, b - a, rows)
        return out

    def close(self):
        for source in self.sources.values():
            source.close()


def _ranges(pages, tile):
    """The arena's write ranges [(lo, hi)]: CHUNK_BYTES each; a tail under
    RANGE_MIN_BYTES moved back to end at the arena's end with that many bytes (its
    first pages written twice); an arena under it: one range, the whole tensor."""
    per = max(1, CHUNK_BYTES // tile)
    least = -(-RANGE_MIN_BYTES // tile)
    out = []
    for lo in range(0, pages, per):
        hi = min(pages, lo + per)
        if (hi - lo) < least:
            lo = max(0, hi - least)
        out.append((lo, hi))
    return out


def load(
    session,
    names=None,
    islands=None,
    placement=PLACEMENT,
    cache=CACHE,
    verify="sample",
    resume=False,
    threads=None,
    log=print,
):
    """Writes the picked weights (`names`: None, names or a predicate) onto a session's
    islands (`islands`: {placement island: session island}, default p mod islands);
    returns Weights. verify "sample"/"full"/None, resume, threads: README.md."""
    with open(placement) as f:
        p = json.load(f)
    tile_bytes = p["tile_bytes"]
    count = _islands_of(session)
    picked = _pick(p["entries"], names)
    if count != len(p["cards"]) and names is None:
        raise ValueError(
            f"a session of {count} islands does not hold the whole backbone: give "
            "names= (and islands= if the default p -> p mod islands does not suit)"
        )
    if islands is None:
        islands = {e["island"]: e["island"] % count for e in picked}
    W = Weights()
    t_start = time.time()
    # each arena: the source runs it copies (cache island, cache first, pages) in order
    plan = {}
    for e in picked:
        key = (islands[e["island"]], e["dtype"])
        plan.setdefault(key, []).append(e)
    whole = names is None and count == len(p["cards"])
    copies = {}
    for (island, dtype), entries in sorted(plan.items()):
        if whole:  # the placement's own arena: pages as placed
            pages = next(
                a["pages"]
                for a in p["arenas"]
                if a["island"] == island and a["dtype"] == dtype
            )
            runs = [(island, 0, pages, 0)]
            firsts = {e["name"]: e["first"] for e in entries}
        else:  # the picked weights, one after another
            runs, firsts, at = [], {}, 0
            for e in entries:
                runs.append((e["island"], e["first"], e["pages"], at))
                firsts[e["name"]] = at
                at += e["pages"]
            pages = at
        arena = _arena(session, island, pages, dtype)
        W.arenas[(island, dtype)] = arena
        copies[(island, dtype)] = runs
        for e in entries:
            W[e["name"]] = Handle(
                e["name"], arena, firsts[e["name"]], e["pages"], e["shape"], dtype,
                island, e["card"],
            )  # fmt: skip
    t_alloc = time.time() - t_start
    total = sum(
        pages * tile_bytes[dtype]
        for (island, dtype), runs in copies.items()
        for _, _, pages, _ in runs
    )
    log(
        f"weights: {len(picked)} weights, {len(W.arenas)} arenas, {total:,} B; "
        f"arenas made in {t_alloc:.2f} s"
    )
    stats = {"written": 0, "skipped": 0}
    lock = threading.Lock()

    def write_island(island):
        t0, written, skipped = time.time(), 0, 0
        for (isl, dtype), runs in copies.items():
            if isl != island:
                continue
            tile = tile_bytes[dtype]
            arena = W.arenas[(isl, dtype)]
            one, part = _part(session, arena, isl)
            gather = _Gather(cache, dtype, tile, runs)
            try:
                ranges = _ranges(gather.pages, tile)
                with ThreadPoolExecutor(1) as ahead:  # the next range read meanwhile
                    coming = ahead.submit(gather.rows, *ranges[0]) if ranges else None
                    for k, (lo, hi) in enumerate(ranges):
                        rows = coming.result()
                        if k + 1 < len(ranges):
                            coming = ahead.submit(gather.rows, *ranges[k + 1])
                        if resume and _holds(one, part, lo, rows):
                            skipped += (hi - lo) * tile
                            continue
                        one.write(part, rows, first=lo)
                        written += (hi - lo) * tile
            finally:
                gather.close()
        sec = time.time() - t0
        with lock:
            stats["written"] += written
            stats["skipped"] += skipped
        log(
            f"weights: island {island}: wrote {written:,} B in {sec:.1f} s "
            f"({written / max(sec, 1e-9) / 1e9:.2f} GB/s), skipped {skipped:,} B"
        )
        return sec

    targets = sorted({isl for isl, _ in copies})
    # islands at once, a run loaded or not (whole box: 31 threads 333.9 s, 8 threads
    # 215.4 s; README.md "The loader")
    n_threads = threads if threads is not None else min(THREADS, len(targets))
    t0 = time.time()
    if n_threads <= 1:
        for island in targets:
            write_island(island)
    else:
        with ThreadPoolExecutor(n_threads) as pool:
            list(pool.map(write_island, targets))
    t_write = time.time() - t0
    W.report = {
        "weights": len(picked),
        "arenas": len(W.arenas),
        "bytes": total,
        "alloc_s": t_alloc,
        "write_s": t_write,
        **stats,
    }
    log(
        f"weights: written {stats['written']:,} B, skipped {stats['skipped']:,} B in "
        f"{t_write:.1f} s ({stats['written'] / max(t_write, 1e-9) / 1e9:.2f} GB/s)"
    )
    if verify:
        W.report["verify"] = check(
            session, W, p, copies, cache, verify, log, threads=n_threads
        )
    return W


def _holds(one, part, first, rows):
    """Whether pages `first` and `first + len(rows) - 1` already hold rows' bytes."""
    for i in (0, len(rows) - 1):
        (got,) = one.read(part, first=first + i, pages=1)
        if not np.array_equal(got.reshape(-1)[: rows.shape[1]], rows[i]):
            return False
    return True


def check(session, W, p, copies, cache, mode="sample", log=print, threads=1):
    """Reads back pages of every weight (or every page) and compares them with the
    cache, an island a thread: {"pages": n, "bad": [(name, page)...], "seconds": s}."""
    tile_bytes = p["tile_bytes"]
    t0 = time.time()
    by_arena = {}
    for h in W.values():
        by_arena.setdefault((h.island, h.dtype), []).append(h)

    def check_island(island):
        checked, bad = 0, []
        for (isl, dtype), handles in sorted(by_arena.items()):
            if isl != island:
                continue
            tile = tile_bytes[dtype]
            one, part = _part(session, W.arenas[(isl, dtype)], isl)
            gather = _Gather(cache, dtype, tile, copies[(isl, dtype)])
            try:
                if mode == "full":
                    per = max(1, CHUNK_BYTES // tile)
                    for lo in range(0, gather.pages, per):
                        hi = min(gather.pages, lo + per)
                        want = gather.rows(lo, hi)
                        (got,) = one.read(part, first=lo, pages=hi - lo)
                        got = got.reshape(hi - lo, -1)[:, :tile]
                        same = np.all(got == want, axis=1)
                        checked += hi - lo
                        bad += [
                            (f"{isl}/{dtype}", int(lo + i))
                            for i in np.nonzero(~same)[0][:8]
                        ]
                    continue
                for h in handles:
                    last = h.first + h.pages - 1
                    for page in sorted({h.first, (h.first + last) // 2, last}):
                        want = gather.rows(page, page + 1)[0]
                        (got,) = one.read(part, first=page, pages=1)
                        checked += 1
                        if not np.array_equal(got.reshape(-1)[:tile], want):
                            bad.append((h.name, page))
            finally:
                gather.close()
        return checked, bad

    islands = sorted({isl for isl, _ in by_arena})
    if threads <= 1:
        results = [check_island(i) for i in islands]
    else:
        with ThreadPoolExecutor(threads) as pool:
            results = list(pool.map(check_island, islands))
    checked = sum(r[0] for r in results)
    bad = [b for r in results for b in r[1]]
    sec = time.time() - t0
    log(
        f"weights: verify ({mode}): {checked:,} pages read back, {len(bad)} differ "
        f"from the cache, {sec:.1f} s"
    )
    return {"pages": checked, "bad": bad, "seconds": sec}
