"""The plan of an attention call on persistent kernels, as pure functions: rows per job
and keys per block under the L1 budget, the broadcast length, every core's jobs, the job
tables. The rules are Kohaku-TT's ktt_flash host program's; sizes are in 32 x 32 tiles
of bf16 (2048 bytes)."""

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
SUB_CHUNK = K["kSubChunk"]
# L1 bytes per core the program's own buffers may take (the matrix product's measured
# budget: 1,381,376 ran on the card, 1,459,200 was rejected at build).
L1_BUDGET = 1_380_000
WINDOW = 2  # K / V blocks in flight per broadcast, as ktt_flash's double buffer
MAX_JOBS = 46  # ktt_flash's limit per core (its runtime arguments)
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

    def last_key(self, row):
        """The last key tile query tile row `row` sees."""
        return min(row, self.skt - 1) if self.causal else self.skt - 1


def dims(batches, heads, kv_heads, query_rows, key_rows, head_dim, value_dim, causal):
    """Dims of q [batches, heads, query_rows, head_dim], k [batches, kv_heads,
    key_rows, head_dim], v [batches, kv_heads, key_rows, value_dim]; lengths and
    head dims must be whole tiles, heads a multiple of kv_heads."""
    if (
        any(x % TILE for x in (query_rows, key_rows, head_dim, value_dim))
        or heads % kv_heads
    ):
        raise ValueError(
            "not whole tiles or heads: "
            f"{(batches, heads, kv_heads, query_rows, key_rows, head_dim, value_dim)}"
        )
    return Dims(
        batches,
        heads,
        kv_heads,
        query_rows // TILE,
        key_rows // TILE,
        head_dim // TILE,
        value_dim // TILE,
        causal,
    )


def block_pages(call, block_keys):
    """Pages of one K / V block: K [dt][block_keys], then V [block_keys][dvt] (the head
    dims of Q K and of V in tiles)."""
    return call.dt * block_keys + block_keys * call.dvt


def buffers(call, rows, block_keys, window=WINDOW):
    """(index, format, page bytes, pages) of the program's buffers."""
    accumulator_columns = call.dvt + 1
    return [
        (K["kCbQ"], "bfloat16", TILE_BYTES, rows * call.dt),
        (K["kCbKv"], "bfloat16", TILE_BYTES, window * block_pages(call, block_keys)),
        (K["kCbOnes"], "bfloat16", TILE_BYTES, 1),
        (K["kCbP"], "bfloat16", TILE_BYTES, 2 * SUB_CHUNK),
        (K["kCbSt"], "float32", TILE32_BYTES, 2 * rows),
        (K["kCbO"], "float32", TILE32_BYTES, rows * accumulator_columns),
        (K["kCbStok"], "bfloat16", TOKEN_BYTES, 4),
        (K["kCbOtok"], "bfloat16", TOKEN_BYTES, rows + 4),
        (K["kCbCausal"], "bfloat16", TILE_BYTES, 1),
        (K["kCbOut"], "bfloat16", TILE_BYTES, 2 * call.dvt),
        (K["kCbJobs"], "uint32", K["kTableBytes"], 1),
        (K["kCbWriterJobs"], "uint32", K["kTableBytes"], 1),
        (K["kCbProf"], "uint32", K["kProfBytes"], 1),
    ]


def l1_bytes(call, rows, block_keys, window=WINDOW):
    return sum(
        page * pages for _, _, page, pages in buffers(call, rows, block_keys, window)
    )


def blocks(call, budget=L1_BUDGET):
    """Rows per job and keys per block as ktt_flash fits them: from (13, 12), keys down
    to 6, then rows down to 4, then keys, then rows, until the buffers fit `budget`;
    (0, 0) when nothing fits."""
    rows, block_keys = 13, 12
    while l1_bytes(call, rows, block_keys) > budget:
        if block_keys > 6:
            block_keys -= SUB_CHUNK
        elif rows > 4:
            rows -= 1
        elif block_keys > SUB_CHUNK:
            block_keys -= SUB_CHUNK
        elif rows > 1:
            rows -= 1
        else:
            return 0, 0
    return rows, block_keys


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


def causal_cuts(rows, chunks):
    """Row bounds of n chunks of about equal area of the causal triangle."""
    return sorted(
        {round(rows * math.sqrt(chunk / chunks)) for chunk in range(chunks + 1)}
    )


def widest(bounds):
    return max(stop - start for start, stop in zip(bounds, bounds[1:]))


def broadcast_chunks(call, rows, broadcasts, length, chunks=None):
    """Per broadcast, its query chunks (batch, head, K / V head, first row, rows): even
    ranges of the (batch, head, row) order, or (causal) chunks of each head (even ranges
    of at most length x rows rows, or `chunks` of equal area), largest work first onto
    the least loaded broadcast below MAX_JOBS chunks."""
    group = call.h // call.hk
    per_broadcast = [[] for _ in range(broadcasts)]
    if not call.causal:
        total = call.b * call.h * call.sqt
        for broadcast in range(broadcasts):
            low = broadcast * total // broadcasts
            high = (broadcast + 1) * total // broadcasts
            position = low
            while position < high:
                head, first_row = divmod(position, call.sqt)
                count = min(high - position, call.sqt - first_row, length * rows)
                batch, query_head = divmod(head, call.h)
                per_broadcast[broadcast].append(
                    (batch, query_head, query_head // group, first_row, count)
                )
                position += count
        return per_broadcast
    if chunks is None:  # even ranges of at most length x rows rows
        chunk_count = math.ceil(call.sqt / (length * rows))
        bounds = [chunk * call.sqt // chunk_count for chunk in range(chunk_count + 1)]
    else:  # `chunks` of about equal area of the triangle
        bounds = causal_cuts(call.sqt, chunks)
        if widest(bounds) > length * rows:
            raise ValueError(
                f"{chunks} chunks per head: one over {length} x {rows} rows"
            )
    pieces = []
    for head in range(call.b * call.h):
        batch, query_head = divmod(head, call.h)
        for start, stop in zip(bounds, bounds[1:]):
            if chunks is None:  # about the chunk's work per core
                work = (
                    math.ceil((stop - start) / length)
                    * (call.last_key(start) + call.last_key(stop - 1) + 2)
                    // 2
                )
            else:  # the chunk's work on its busiest core (rows `length` apart)
                work = max(
                    sum(
                        call.last_key(row) + 1 for row in range(start + y, stop, length)
                    )
                    for y in range(length)
                )
            pieces.append(
                (work, (batch, query_head, query_head // group, start, stop - start))
            )
    pieces.sort(key=lambda piece: -piece[0])  # stable: equal work keeps its order
    load = [0] * broadcasts
    for work, chunk in pieces:
        open_broadcasts = [
            broadcast
            for broadcast in range(broadcasts)
            if len(per_broadcast[broadcast]) < MAX_JOBS
        ]
        target = (
            min(open_broadcasts, key=lambda broadcast: load[broadcast])
            if open_broadcasts
            else min(range(broadcasts), key=lambda broadcast: load[broadcast])
        )
        per_broadcast[target].append(chunk)
        load[target] += work
    return per_broadcast


def plan(call, rows, grid, length, chunks=None):
    """Jobs per core (index y * grid_columns + x) for broadcasts of `length` cores down
    each grid column, the sender at the top; each core takes its partition of a
    broadcast job's rows: contiguous, or (causal) `length` apart."""
    grid_columns, grid_rows = grid
    broadcasts = [
        (x, top)
        for x in range(grid_columns)
        for top in range(0, grid_rows - length + 1, length)
    ]
    by_broadcast = broadcast_chunks(call, rows, len(broadcasts), length, chunks)
    per_core = [[] for _ in range(grid_columns * grid_rows)]
    for (x, top), broadcast in zip(broadcasts, by_broadcast):
        for y in range(top, top + length):
            for batch, query_head, kv_head, first_row, count in broadcast:
                if call.causal:
                    start = first_row + (y - top)
                    job_rows = (
                        math.ceil((first_row + count - start) / length)
                        if start < first_row + count
                        else 0
                    )
                    start, stride = min(start, first_row + count), length
                else:
                    rows_per_core = math.ceil(count / length)
                    start = min(
                        first_row + (y - top) * rows_per_core, first_row + count
                    )
                    job_rows, stride = min(rows_per_core, first_row + count - start), 1
                query_row = (batch * call.h + query_head) * call.sqt + start
                kv_index = batch * call.hk + kv_head
                per_core[y * grid_columns + x].append(
                    Job(
                        query_row * call.dt,
                        kv_index * call.skt * call.dt,
                        kv_index * call.skt * call.dvt,
                        query_row * call.dvt,
                        job_rows,
                        start,
                        call.last_key(first_row + count - 1) + 1,
                        stride,
                    )
                )
    return per_core


def score_tiles(call, jobs):
    """Score tiles of one core's jobs."""
    return sum(
        call.last_key(job.start + row * job.stride) + 1
        for job in jobs
        for row in range(job.rows)
    )


def choose_broadcasts(call, rows, grid, lengths=None, chunks=None):
    """(length, jobs per core): the fewest score tiles on the busiest core, then the
    longest broadcast, among lengths dividing the grid height with MAX_JOBS jobs or
    fewer."""
    best = None
    for length in lengths or range(1, grid[1] + 1):
        if grid[1] % length:
            continue
        try:
            per_core = plan(call, rows, grid, length, chunks)
        except ValueError:
            continue
        tiles = max(score_tiles(call, jobs) for jobs in per_core)
        if max(len(jobs) for jobs in per_core) <= MAX_JOBS and (
            best is None or tiles <= best[0]
        ):
            best = (tiles, length, per_core)
    if best is None:
        raise ValueError(f"no broadcast length keeps {MAX_JOBS} jobs per core")
    return best[1], best[2]


@dataclass(frozen=True)
class Plan:
    """A call's plan: dims, rows per job (r), keys per block (kb), window, broadcast
    length, and every core's jobs."""

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

    def blocks_per_broadcast(self, core):
        """K / V blocks the broadcast of `core` delivers in one call."""
        return sum(math.ceil(job.nk / self.kb) for job in self.jobs[core])

    def summary(self):
        return {
            "r": self.r,
            "kb": self.kb,
            "window": self.window,
            "broadcast_length": self.length,
            "jobs_max": max(len(jobs) for jobs in self.jobs),
            "score_tiles_max": max(score_tiles(self.dims, jobs) for jobs in self.jobs),
            "l1_bytes": self.l1_bytes(),
        }


def make(
    call,
    grid,
    budget=L1_BUDGET,
    rows=None,
    block_keys=None,
    length=None,
    chunks=None,
    lengths=None,
):
    """The plan of dims `call` on a grid (columns, rows): rows per job and keys per
    block fitted to `budget` (or as given), the broadcast length chosen (among
    `lengths`, or as given), causal chunks per head as given."""
    fit_rows, fit_block_keys = blocks(call, budget)
    rows, block_keys = rows or fit_rows, block_keys or fit_block_keys
    if rows == 0 or l1_bytes(call, rows, block_keys) > budget:
        raise ValueError(
            f"buffers of {rows} rows, {block_keys} keys do not fit {budget} bytes"
        )
    length, jobs = choose_broadcasts(
        call, rows, grid, [length] if length else lengths, chunks
    )
    return Plan(call, rows, block_keys, WINDOW, length, jobs)


def tables(plan):
    """The job tables: one kTableBytes page of uint32 words per core, [cores, words]."""
    words_per_table = K["kTableBytes"] // 4
    table_words = np.zeros((len(plan.jobs), words_per_table), np.uint32)
    for core, jobs in enumerate(plan.jobs):
        if len(jobs) > K["kJobsMax"]:
            raise ValueError(f"core {core}: {len(jobs)} jobs > {K['kJobsMax']}")
        table_words[core, K["kTabJobs"]] = len(jobs)
        for job_index, job in enumerate(jobs):
            first_word = K["kTabFirst"] + K["kJobWords"] * job_index
            table_words[core, first_word : first_word + K["kJobWords"]] = [
                job.q0,
                job.k0,
                job.v0,
                job.o0,
                job.rows,
                job.start,
                job.nk,
                job.stride,
            ]
    return table_words


def float_bits(value):
    """The bits of `value` as an fp32."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def exp_constants(scale, approx=False):
    """(scale bits, threshold bits) as fp32 for the compute kernel: the scale times
    log2 e, and the row reference's threshold (ktt_flash's tau 16, accurate or
    approximate units)."""
    scaled = scale * 1.4426950408889634
    threshold = 32500.818359375 + 256.0 * 16.0 if approx else 639.0 + 16.0
    return float_bits(scaled), float_bits(threshold)


def call_args(
    q_address,
    k_address,
    v_address,
    o_address,
    job_tables,
    mode=K["kModeFull"],
    passes=1,
    first_job=0,
    job_count=0,
):
    """Command words a0..a8 of ATTEND: jobs [first_job, first_job + job_count) of every
    core's table (job_count 0: to the end)."""
    return [
        q_address,
        k_address,
        v_address,
        o_address,
        job_tables,
        mode,
        passes,
        first_job,
        job_count,
    ]


def flops(call):
    """Floating-point operations of one call: the score and the P V products (multiply
    and add counted) over the tiles the mask leaves, in 32 x 32 tiles."""
    rows = [call.last_key(row) + 1 for row in range(call.sqt)]
    return 2 * call.b * call.h * sum(rows) * (call.dt + call.dvt) * TILE**3
