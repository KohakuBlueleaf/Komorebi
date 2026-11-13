"""The plan of an attention call on resident workers, as pure functions: rows per job R
and key block KB under the L1 budget, the lane length, every core's jobs, the job
tables. The rules are Kohaku-TT's ktt_flash host program's; sizes are in 32 x 32
tiles of bf16 (2048 bytes)."""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from komorebi import abi

HERE = Path(__file__).resolve().parent
ATTENTION_H = HERE / "kernels" / "attention.h"
K = abi.constants(ATTENTION_H, known=abi.headers().constants)

TILE = 32
TILE_BYTES = K["kTileBytes"]
TILE32_BYTES = K["kTile32Bytes"]
TOKEN_BYTES = K["kTokenBytes"]
SC = K["kSubChunk"]
# L1 bytes per core the program's own buffers may take (the matrix product's measured
# budget: 1,381,376 ran on the card, 1,459,200 was refused at build).
L1_BUDGET = 1_380_000
WINDOW = 2  # K / V blocks in flight per lane, as ktt_flash's double buffer
MAX_JOBS = 46  # ktt_flash's limit per core (its runtime arguments)
LANE_PAGES_MAX = 65535  # pages through one lane in one record (Plan.passes_max)
# The benchmark cases: (B, H, Hk, S, D, causal).
SHAPES = {
    "t1": (1, 1, 1, 128, 64, False),
    "t2": (1, 4, 4, 512, 64, False),
    "t3": (1, 2, 1, 256, 128, True),
    "long64": (1, 32, 32, 4096, 64, False),
    "wide128": (1, 24, 24, 4096, 128, False),
    "short64": (8, 16, 16, 256, 64, False),
    "causal64": (1, 8, 8, 1024, 64, True),
    "gqa128": (2, 8, 2, 1024, 128, False),
}


@dataclass(frozen=True)
class Dims:
    """The call in tiles: batches, query heads, K / V heads, query and key tile rows,
    head dims of Q K and of V, causal (top-left: query row i sees keys <= i)."""

    b: int
    h: int
    hk: int
    sqt: int
    skt: int
    dt: int
    dvt: int
    causal: bool

    def last_key(self, i):
        """The last key tile query tile row i sees."""
        return min(i, self.skt - 1) if self.causal else self.skt - 1


def dims(b, h, hk, sq, sk, d, dv, causal):
    """Dims of q [b, h, sq, d], k [b, hk, sk, d], v [b, hk, sk, dv]; lengths and head
    dims must be whole tiles, h a multiple of hk."""
    if any(x % TILE for x in (sq, sk, d, dv)) or h % hk:
        raise ValueError(f"not whole tiles or heads: {(b, h, hk, sq, sk, d, dv)}")
    return Dims(b, h, hk, sq // TILE, sk // TILE, d // TILE, dv // TILE, causal)


def block_pages(s, kb):
    """Pages of one K / V block: K [DT][KB], then V [KB][DV]."""
    return s.dt * kb + kb * s.dvt


def buffers(s, r, kb, window=WINDOW):
    """(index, format, page bytes, pages) of the program's buffers."""
    w = s.dvt + 1
    return [
        (K["kCbQ"], "bfloat16", TILE_BYTES, r * s.dt),
        (K["kCbKv"], "bfloat16", TILE_BYTES, window * block_pages(s, kb)),
        (K["kCbOnes"], "bfloat16", TILE_BYTES, 1),
        (K["kCbP"], "bfloat16", TILE_BYTES, 2 * SC),
        (K["kCbSt"], "float32", TILE32_BYTES, 2 * r),
        (K["kCbO"], "float32", TILE32_BYTES, r * w),
        (K["kCbStok"], "bfloat16", TOKEN_BYTES, 4),
        (K["kCbOtok"], "bfloat16", TOKEN_BYTES, r + 4),
        (K["kCbCausal"], "bfloat16", TILE_BYTES, 1),
        (K["kCbOut"], "bfloat16", TILE_BYTES, 2 * s.dvt),
        (K["kCbJobs"], "uint32", K["kTableBytes"], 1),
        (K["kCbWriterJobs"], "uint32", K["kTableBytes"], 1),
        (K["kCbProf"], "uint32", K["kProfBytes"], 1),
    ]


def l1_bytes(s, r, kb, window=WINDOW):
    return sum(page * pages for _, _, page, pages in buffers(s, r, kb, window))


def blocks(s, budget=L1_BUDGET):
    """R and KB as ktt_flash fits them: from (13, 12), KB down to 6, then R down to
    4, then KB, then R, until the buffers fit `budget`; (0, 0) when nothing fits."""
    r, kb = 13, 12
    while l1_bytes(s, r, kb) > budget:
        if kb > 6:
            kb -= SC
        elif r > 4:
            r -= 1
        elif kb > SC:
            kb -= SC
        elif r > 1:
            r -= 1
        else:
            return 0, 0
    return r, kb


@dataclass(frozen=True)
class Job:
    """One job of a core: `rows` query tile rows of one head (first row `start` of the
    head, rows `stride` apart) against key tiles [0, nk); first pages of Q, K, V, O."""

    q0: int
    k0: int
    v0: int
    o0: int
    rows: int
    start: int
    nk: int
    stride: int


def lane_chunks(s, r, lanes, length):
    """Per lane, its query chunks (batch, head, K / V head, first row, rows): even
    ranges of the (batch, head, row) order, or (causal) chunks largest work first
    onto the least loaded lane below MAX_JOBS chunks."""
    group = s.h // s.hk
    out = [[] for _ in range(lanes)]
    if not s.causal:
        total = s.b * s.h * s.sqt
        for i in range(lanes):
            lo, hi = i * total // lanes, (i + 1) * total // lanes
            pos = lo
            while pos < hi:
                head, r0 = divmod(pos, s.sqt)
                n = min(hi - pos, s.sqt - r0, length * r)
                bi, hi_ = divmod(head, s.h)
                out[i].append((bi, hi_, hi_ // group, r0, n))
                pos += n
        return out
    pieces = []
    for head in range(s.b * s.h):
        bi, hi_ = divmod(head, s.h)
        n_chunks = math.ceil(s.sqt / (length * r))
        for c in range(n_chunks):
            r0, r1 = c * s.sqt // n_chunks, (c + 1) * s.sqt // n_chunks
            work = (
                math.ceil((r1 - r0) / length)
                * (s.last_key(r0) + s.last_key(r1 - 1) + 2)
                // 2
            )
            pieces.append((work, (bi, hi_, hi_ // group, r0, r1 - r0)))
    pieces.sort(key=lambda p: -p[0])  # stable: equal work keeps its order
    load = [0] * lanes
    for work, chunk in pieces:
        open_ = [j for j in range(lanes) if len(out[j]) < MAX_JOBS]
        i = (
            min(open_, key=lambda j: load[j])
            if open_
            else min(range(lanes), key=lambda j: load[j])
        )
        out[i].append(chunk)
        load[i] += work
    return out


def plan(s, r, grid, length):
    """Jobs per core (index y * gx + x) for lanes of `length` cores down each grid
    column, the sender at the top; each core takes its share of a lane job's rows:
    contiguous, or (causal) `length` apart."""
    gx, gy = grid
    lanes = [(x, y0) for x in range(gx) for y0 in range(0, gy - length + 1, length)]
    chunks = lane_chunks(s, r, len(lanes), length)
    per_core = [[] for _ in range(gx * gy)]
    for (x, y0), lane in zip(lanes, chunks):
        for y in range(y0, y0 + length):
            for bi, hi, g, r0, n in lane:
                if s.causal:
                    start = r0 + (y - y0)
                    rows = math.ceil((r0 + n - start) / length) if start < r0 + n else 0
                    start, stride = min(start, r0 + n), length
                else:
                    per = math.ceil(n / length)
                    start = min(r0 + (y - y0) * per, r0 + n)
                    rows, stride = min(per, r0 + n - start), 1
                qrow = (bi * s.h + hi) * s.sqt + start
                kv = bi * s.hk + g
                per_core[y * gx + x].append(
                    Job(
                        qrow * s.dt,
                        kv * s.skt * s.dt,
                        kv * s.skt * s.dvt,
                        qrow * s.dvt,
                        rows,
                        start,
                        s.last_key(r0 + n - 1) + 1,
                        stride,
                    )
                )
    return per_core


def score_tiles(s, jobs):
    """Score tiles of one core's jobs."""
    return sum(
        s.last_key(j.start + i * j.stride) + 1 for j in jobs for i in range(j.rows)
    )


def choose_lanes(s, r, grid, lengths=None):
    """(length, jobs per core): the fewest score tiles on the busiest core, then the
    longest lane, among lengths dividing the grid height with MAX_JOBS jobs or fewer."""
    best = None
    for length in lengths or range(1, grid[1] + 1):
        if grid[1] % length:
            continue
        p = plan(s, r, grid, length)
        tiles = max(score_tiles(s, jobs) for jobs in p)
        if max(len(jobs) for jobs in p) <= MAX_JOBS and (
            best is None or tiles <= best[0]
        ):
            best = (tiles, length, p)
    if best is None:
        raise ValueError(f"no lane length keeps {MAX_JOBS} jobs per core")
    return best[1], best[2]


@dataclass(frozen=True)
class Plan:
    """A call's plan: dims, R, KB, window, lane length, and every core's jobs."""

    dims: Dims
    r: int
    kb: int
    window: int
    length: int
    jobs: list

    def block_pages(self):
        return block_pages(self.dims, self.kb)

    def buffers(self):
        return buffers(self.dims, self.r, self.kb, self.window)

    def l1_bytes(self):
        return l1_bytes(self.dims, self.r, self.kb, self.window)

    def blocks_per_lane(self, core):
        """K / V blocks the lane of `core` hands over in one call."""
        return sum(math.ceil(j.nk / self.kb) for j in self.jobs[core])

    def passes_max(self):
        """Passes one record may make: a lane's sender reads its buffer's 16-bit popped
        counter only at the record's end, so fewer than 65536 pages per record."""
        most = max(self.blocks_per_lane(c) for c in range(len(self.jobs)))
        return max(1, LANE_PAGES_MAX // max(1, most * self.block_pages()))

    def summary(self):
        return {
            "r": self.r,
            "kb": self.kb,
            "window": self.window,
            "lane_length": self.length,
            "jobs_max": max(len(j) for j in self.jobs),
            "score_tiles_max": max(score_tiles(self.dims, j) for j in self.jobs),
            "l1_bytes": self.l1_bytes(),
        }


def make(s, grid, budget=L1_BUDGET, r=None, kb=None, length=None):
    """The plan of dims `s` on a grid (gx, gy): R and KB fitted to `budget` (or as
    given), the lane length chosen (or as given)."""
    fit_r, fit_kb = blocks(s, budget)
    r, kb = r or fit_r, kb or fit_kb
    if r == 0 or l1_bytes(s, r, kb) > budget:
        raise ValueError(f"buffers of R {r}, KB {kb} do not fit {budget} bytes")
    length, jobs = choose_lanes(s, r, grid, [length] if length else None)
    p = Plan(s, r, kb, WINDOW, length, jobs)
    most = max(p.blocks_per_lane(c) for c in range(len(jobs))) * p.block_pages()
    if most > LANE_PAGES_MAX:
        raise ValueError(f"{most} pages through a lane in one call (Plan.passes_max)")
    return p


def tables(p):
    """The job tables: one kTableBytes page of uint32 words per core, [cores, words]."""
    words = K["kTableBytes"] // 4
    out = np.zeros((len(p.jobs), words), np.uint32)
    for core, jobs in enumerate(p.jobs):
        if len(jobs) > K["kJobsMax"]:
            raise ValueError(f"core {core}: {len(jobs)} jobs > {K['kJobsMax']}")
        out[core, K["kTabJobs"]] = len(jobs)
        for i, j in enumerate(jobs):
            at = K["kTabFirst"] + K["kJobWords"] * i
            out[core, at : at + K["kJobWords"]] = [
                j.q0,
                j.k0,
                j.v0,
                j.o0,
                j.rows,
                j.start,
                j.nk,
                j.stride,
            ]
    return out


def float_bits(x):
    """The bits of x as an fp32."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def exp_constants(scale, approx=False):
    """(A, threshold) as fp32 bits for the compute kernel: A = scale log2 e, and the
    row reference's threshold (ktt_flash's tau 16, accurate or approximate units)."""
    a = scale * 1.4426950408889634
    thr = 32500.818359375 + 256.0 * 16.0 if approx else 639.0 + 16.0
    return float_bits(a), float_bits(thr)


def call_args(q, k, v, o, table, mode=K["kModeFull"], passes=1):
    """Record words a0..a6 of ATTEND."""
    return [q, k, v, o, table, mode, passes]


def flops(s):
    """Floating-point operations of one call: the score and the P V products (multiply
    and add counted) over the tiles the mask leaves, in 32 x 32 tiles."""
    rows = [s.last_key(i) + 1 for i in range(s.sqt)]
    return 2 * s.b * s.h * sum(rows) * (s.dt + s.dvt) * TILE**3
