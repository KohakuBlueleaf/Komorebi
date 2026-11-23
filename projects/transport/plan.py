"""The host's side of one apply y = M x of a hierarchical transport matrix: the set
read, the receiver tiles dealt to the cores, w's pages laid out, each core's table,
the sizes a core needs and the apply emulated. Sizes are in 32 x 32 tiles; a weight
tile is bfp8_b (1,088 B)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from komorebi import formats

TILE = 32
WEIGHT_TILE_BYTES = formats.tile_bytes("bfp8_b")
X_TILE_BYTES = formats.tile_bytes("bfloat16")
WORDS = 8  # an entry's words (Slice in transport.py)
HEAD = 8  # a table page's first words: its count, then the core's own range
PAGE_WORDS = 1024  # a table page: 4 KiB
ENTRIES_A_PAGE = (PAGE_WORDS - HEAD) // WORDS


@dataclass
class Set:
    """One set's files: the blocks (in tiles), the weights and the vectors."""

    folder: Path
    row0: np.ndarray  # receiver tiles row0 : row1 of each block
    row1: np.ndarray
    col0: np.ndarray  # source tiles col0 : col1
    col1: np.ndarray
    rank: np.ndarray  # rank tiles R (0 for a dense block)
    offset_a: np.ndarray  # numbers into weights.f32
    offset_b: np.ndarray
    tiles: int  # receiver (and source) tiles of the matrix

    @property
    def blocks(self):
        return len(self.row0)

    def weights(self):
        """weights.f32 as float32 numbers, mapped."""
        return np.memmap(self.folder / "weights.f32", np.float32, "r")

    def vector(self, name):
        """x.npy, y_float.npy, ...: [N, 3] float32."""
        return np.load(self.folder / name)


def load(folder):
    """The set in `folder` (blocks.npz and the rest, as its README.md lays them out)."""
    folder = Path(folder)
    b = np.load(folder / "blocks.npz")
    if not np.all(b["scale"] == 1.0):
        raise ValueError(f"{folder}: a block's scale is not 1 (not applied here)")
    tiles = len(np.load(folder / "perm.npy")) // TILE
    return Set(
        folder,
        b["row0"] // TILE,
        b["row1"] // TILE,
        b["col0"] // TILE,
        b["col1"] // TILE,
        b["rank_padded"] // TILE * (b["kind"] == 1),
        b["offset_a"],
        b["offset_b"],
        tiles,
    )


def weight_pages(s):
    """Each block's first page in w and the pages in all (with one zero tile at the
    end). A dense block's K x N tiles column by column (tile (k, j) at page j K + k); a
    low-rank block's V (K x R) column by column, then its U^T (R x N) so."""
    k = s.col1 - s.col0
    n = s.row1 - s.row0
    size = np.where(s.rank == 0, k * n, k * s.rank + s.rank * n)
    first = np.concatenate([[0], np.cumsum(size)])
    return first[:-1], int(first[-1]) + 1


def row_blocks(s):
    """For each receiver tile, the blocks whose rows hold it: (start [T + 1],
    blocks)."""
    n = s.row1 - s.row0
    blocks = np.repeat(np.arange(s.blocks), n)
    rows = np.concatenate([np.arange(a, b) for a, b in zip(s.row0, s.row1)])
    order = np.argsort(rows, kind="stable")
    start = np.searchsorted(rows[order], np.arange(s.tiles + 1))
    return start, blocks[order]


def deal(s, cores):
    """Contiguous ranges of receiver tiles, one a core, the largest cost least: a
    range's cost is the weight tiles its slices read (a dense slice K n, a low-rank
    one its own V, K R, and its U^T columns, R n). Returns first [cores + 1]."""
    start, blocks = row_blocks(s)
    k = s.col1 - s.col0
    per_row = np.where(s.rank == 0, k, s.rank)  # a receiver's tiles of each block

    def ranges(cap):
        firsts, a, cost, touched = [0], 0, 0, set()
        for r in range(s.tiles):
            here = blocks[start[r] : start[r + 1]]
            add = int(per_row[here].sum())
            add += sum(int(k[q] * s.rank[q]) for q in here if q not in touched)
            if cost + add > cap and r > a:
                firsts.append(r)
                if len(firsts) > cores:
                    return None
                a, cost, touched = r, 0, set()
                add = int(per_row[here].sum()) + int((k[here] * s.rank[here]).sum())
            cost += add
            touched.update(int(q) for q in here if s.rank[q])
        return firsts + [s.tiles] * (cores + 1 - len(firsts))

    low, high = 1, int((k * (s.row1 - s.row0) + k * s.rank).sum())
    while low < high:
        cap = (low + high) // 2
        if ranges(cap) is None:
            low = cap + 1
        else:
            high = cap
    return np.array(ranges(low))


@dataclass
class Tables:
    """Each core's entries ([cores] lists of 8-word rows), its range of receiver tiles
    and its cost; the table tensor's words [cores x pages, PAGE_WORDS]."""

    first: np.ndarray  # [cores + 1] receiver tiles
    entries: list
    pages: int  # table pages a core
    words: np.ndarray
    tiles_read: np.ndarray  # [cores] weight tiles
    x_read: np.ndarray  # [cores] x tiles (each entry its source range)
    zero_entries: int


def tables(s, first, zero_page):
    """Each core's entries, one a block slice (the block's receivers it owns), by the
    block's first source tile, then its rows; words as transport.Slice's (README.md,
    "Tables"). A receiver no block reaches gets a zero tile's entry, so that every sum
    is written once."""
    pages_of, _ = weight_pages(s)
    start, blocks = row_blocks(s)
    cores = len(first) - 1
    out, tiles_read, x_read, zeros = [], [], [], 0
    for c in range(cores):
        a, b = int(first[c]), int(first[c + 1])
        mine = np.unique(blocks[start[a] : start[b]])
        mine = mine[np.lexsort((s.row0[mine], s.col0[mine]))]
        rows, read, xs = [], 0, 0
        touched = np.zeros(b - a, bool)
        for q in mine:
            j0, j1 = max(a, s.row0[q]), min(b, s.row1[q])
            k, rank, n = int(s.col1[q] - s.col0[q]), int(s.rank[q]), int(j1 - j0)
            base, skip = int(pages_of[q]), int(j0 - s.row0[q])
            if rank == 0:
                row = (s.col0[q], k, base + skip * k, k * n, 0, 0, j0 - a, 0)
            else:
                u = base + k * rank + skip * rank
                row = (s.col0[q], k, base, k * rank, u, rank * n, j0 - a, rank)
            rows.append(row)
            read += row[3] + row[5]
            xs += k
            touched[j0 - a : j1 - a] = True
        for j in np.flatnonzero(~touched):
            rows.append((0, 1, zero_page, 1, 0, 0, int(j), 0))
            zeros += 1
        out.append(np.array(rows, np.int64).reshape(-1, WORDS))
        tiles_read.append(read)
        x_read.append(xs)
    pages = max(1, max(-(-len(r) // ENTRIES_A_PAGE) for r in out))
    words = np.zeros((cores * pages, PAGE_WORDS), np.uint32)
    for c, rows in enumerate(out):
        for p in range(pages):
            part = rows[p * ENTRIES_A_PAGE : (p + 1) * ENTRIES_A_PAGE]
            page = words[c * pages + p]
            page[0:3] = (len(part), first[c], first[c + 1] - first[c])
            page[HEAD : HEAD + part.size] = part.reshape(-1)
    return Tables(
        first, out, pages, words, np.array(tiles_read), np.array(x_read), zeros
    )


def weight_tiles(s, start, count):
    """Pages start .. start + count of w as [count, 32, 32] float32 (each tile as the
    product x W takes it: rows the sources, columns the receivers)."""
    pages_of, total = weight_pages(s)
    numbers = s.weights()
    out = np.zeros((count, TILE, TILE), np.float32)
    first = np.searchsorted(pages_of, start, side="right") - 1
    for q in range(max(first, 0), s.blocks):
        base = int(pages_of[q])
        if base >= start + count:
            break
        k, n, rank = (
            int(s.col1[q] - s.col0[q]),
            int(s.row1[q] - s.row0[q]),
            int(s.rank[q]),
        )
        parts = []
        if rank == 0:
            parts.append((int(s.offset_a[q]), k, n))
        else:
            parts.append((int(s.offset_a[q]), k, rank))
            parts.append((int(s.offset_b[q]), rank, n))
        tiles = []
        for offset, rows, cols in parts:
            m = np.asarray(numbers[offset : offset + rows * cols * TILE * TILE])
            m = m.reshape(rows, TILE, cols, TILE).transpose(2, 0, 1, 3)
            tiles.append(m.reshape(-1, TILE, TILE))  # column by column
        tiles = np.concatenate(tiles)
        lo, hi = max(base, start), min(base + len(tiles), start + count)
        if lo < hi:
            out[lo - start : hi - start] = tiles[lo - base : hi - base]
    del total
    return out


def encode_weights(s, cache=None, chunk=1 << 16):
    """w's bytes [pages, 1088] uint8: bfp8_b of tiles stored transposed (the turned
    product's weight), the host encode; kept in `cache` (.npy) when given."""
    if cache is not None and Path(cache).exists():
        return np.load(cache, mmap_mode="r")
    _, total = weight_pages(s)
    raw = np.zeros((total, WEIGHT_TILE_BYTES), np.uint8)
    for start in range(0, total, chunk):
        count = min(chunk, total - start)
        tiles = weight_tiles(s, start, count)
        data, _ = formats.to_bytes(tiles, "bfp8_b", "tile", turned=True)
        raw[start : start + count] = data.reshape(count, WEIGHT_TILE_BYTES)
    if cache is not None:
        np.save(cache, raw)
    return raw


def decode_weights(raw):
    """The values the card multiplies: [pages, 32, 32] float32 as x W takes them."""
    out = np.zeros((len(raw), TILE, TILE), np.float32)
    for start in range(0, len(raw), 1 << 16):
        part = np.asarray(raw[start : start + (1 << 16)])
        values = formats.from_bytes(part, (len(part), TILE, TILE), "bfp8_b", "tile")
        out[start : start + len(part)] = formats.transposed_tiles(values)
    return out


def x_tiles(x):
    """x [N, 3] as the [32, N] row of tiles the card reads (colours as rows 0 to 2)."""
    row = np.zeros((TILE, len(x)), np.float32)
    row[:3] = x.T
    return row


def bf16(values):
    """Rounded to bfloat16 (nearest, ties to even), as float32."""
    return formats.bf16_values(formats.bf16_bits(np.asarray(values, np.float32)))


def emulate(s, w, x):
    """y [N, 3] as the card computes it from the decoded weights w [pages, 32, 32]: x
    and t in bfloat16, every product exact, sums in float64 (the card's in float32)."""
    pages_of, _ = weight_pages(s)
    xr = bf16(x_tiles(x)).astype(np.float64)[:3]  # [3, N]
    y = np.zeros((3, s.tiles * TILE))
    for q in range(s.blocks):
        k, n, rank = (
            int(s.col1[q] - s.col0[q]),
            int(s.row1[q] - s.row0[q]),
            int(s.rank[q]),
        )
        base = int(pages_of[q])
        src = xr[:, s.col0[q] * TILE : s.col1[q] * TILE]
        dst = slice(s.row0[q] * TILE, s.row1[q] * TILE)

        def matrix(at, rows, cols):
            t = w[at : at + rows * cols].reshape(cols, rows, TILE, TILE)
            return t.transpose(1, 2, 0, 3).reshape(rows * TILE, cols * TILE)

        if rank == 0:
            y[:, dst] += src @ matrix(base, k, n)
        else:
            t = bf16(src @ matrix(base, k, rank)).astype(np.float64)
            y[:, dst] += t @ matrix(base + k * rank, rank, n)
    return y.T.astype(np.float32)


def sizes(s, t, kmax_cap=None):
    """The numbers a core's buffers are made for: the most entries, table pages, own
    receiver tiles, x's tiles of one entry, rank tiles."""
    own = np.diff(t.first)
    entries = np.array([len(r) for r in t.entries])
    k = s.col1 - s.col0
    return {
        "cores": len(own),
        "entries_most": int(entries.max()),
        "entries_mean": float(entries.mean()),
        "table_pages": t.pages,
        "own_most": int(own.max()),
        "own_least": int(own.min()),
        "k_most": int(k.max()),
        "rank_most": int(s.rank.max()),
        "tiles_read_most": int(t.tiles_read.max()),
        "tiles_read_mean": float(t.tiles_read.mean()),
        "tiles_read_least": int(t.tiles_read.min()),
        "x_read_most": int(t.x_read.max()),
        "x_read_mean": float(t.x_read.mean()),
        "zero_entries": t.zero_entries,
    }
