"""The layer's attention core: softmax(q k^T scale + bias) v over the key list (the
sink, the window, the selected entries), attention_class's ATTEND with a bias tile under
every score tile (flash_local.hpp). The 64 heads are the query tile rows of one head,
sharing the one key-value broadcast. README.md, "The key list"."""

import numpy as np

import komorebi as km
from projects.attention import card
from projects.attention import plan as pl

HEADS, HEAD_DIM, TILE = 64, 512, 32
SCALE = HEAD_DIM**-0.5
MASKED = -1e30  # the bias of a key a row does not see (the causal tile's value)
MASKS = 5  # a mask variant's tiles: key tiles 1 .. 5 (the window's block)
ENTRY_TILES = 16  # the selected entries' key tiles (512 entries), from key tile 6
KEY_TILES = 1 + MASKS + ENTRY_TILES  # the key list: sink, window, block, entries
WINDOW, RING_TILES = 128, 4  # the window's slots: key tiles 1 .. 4 (window.py)
DECODE_VARIANTS = WINDOW  # a decode token at pos: variant min(pos, 127)
# a block at start s (s % 32 == 0): 128 + s // 32 % 4 (+ 4 if s >= 128)
PREFILL_VARIANTS = 8
# The plan's sizes: Q's and V's tiles a row, query tile rows a job, key tiles a block
# (block 0 the sink's and the window's), the broadcast's length (README.md).
DT, DV, R, KB, LENGTH = 16, 16, 1, 6, 9
# blocks in flight a broadcast (384 KiB each): 1 fits a joined island's L1 beside the
# service overlay's own slot; 2 when that overlay can share the reader's slots
BROADCAST_WINDOW = 1
if KB != MASKS + 1:
    raise ValueError("block 0 of the key list is the sink's and the window's tiles")
SUB = pl.SUB_CHUNK  # key tiles a compute step
QB = R * DT  # a job's block of Q
SCALE_BITS, THRESHOLD_BITS = pl.exp_constants(SCALE)


class Job(km.Entry):
    """A job of the table (projects/attention, attention.h kJob*): first Q, K, V and O
    pages, its rows, the first row in its head, key tiles, the row step."""

    q: int
    k: int
    v: int
    o: int
    rows: int
    start: int
    keys: int
    stride: int


class AttendLocal(
    km.Program,
    compute={
        "math_fidelity": "HiFi2",
        "fp32_dest_acc_en": True,
        "dst_full_sync_en": True,
    },
    startup=km.Startup("ones", "causal", "p", order="reverse"),
    defines={
        "FLASH_HEAD_TILES": DT,
        "FLASH_VALUE_TILES": DV,
        "FLASH_ROWS": R,
        "FLASH_BLOCK_KEYS": KB,
        "FLASH_SUB_CHUNK": SUB,
        "FLASH_SCALE_BITS": SCALE_BITS,
        "FLASH_THRESHOLD_BITS": THRESHOLD_BITS,
    },
):
    """attend: projects/attention's ATTEND (q, K = V = kv, o, the job tables; one pass
    of every job), then the bias: the layer's sink tiles (a query tile row's) and the
    masks' variant (its key tiles 1 .. 5): each core's jobs, Q's rows and their bias
    against the job's K / V blocks from its band's broadcast, each row of O."""

    ones: km.L1[km.bf16, 1, 32, 32] = km.own(push="pack")
    causal: km.L1[km.bf16, 1, 32, 32] = km.own(push="pack")
    p: km.L1[km.bf16, 2 * SUB, 32, 32] = km.own(push="pack")
    state: km.L1[km.fp32, 2 * R, 32, 32] = km.own(push=None)
    acc: km.L1[km.fp32, R * (DV + 1), 32, 32] = km.own(push=None)
    state_token: km.L1[km.bf16, 4, 32, 32] = km.own(push="pack")
    acc_token: km.L1[km.bf16, R + 4, 32, 32] = km.own(push="pack")

    @km.handler
    def attend(
        self,
        q: km.Tensor,
        kv: km.Tensor,
        o: km.Tensor,
        tables: km.Tensor,
        sinks: km.Tensor,
        masks: km.Tensor,
        variant: int,
        entries: km.Tensor,
    ):
        # Each command builds its constant tiles and takes them back at its end: other
        # programs run between commands, and a switch waits for empty buffers.
        km.cxx(
            """
            MATH((llk_math_eltwise_unary_sfpu_init<
                  SfpuType::exponential, DST_ACCUM_MODE>()));
            MATH((flash::init(__builtin_bit_cast(float, FLASH_SCALE_BITS))));
            MATH((flash::macro_init()));
            PACK((flash::addr_init()));
            flash_local::constant_tiles<ones, causal, p, acc>();
            cb_reserve_back(state, 2 * FLASH_ROWS);
            cb_reserve_back(acc, FLASH_ROWS * (FLASH_VALUE_TILES + 1));
            """,
            takes=[
                self.ones,
                self.p,
                self.state,
                self.acc,
                self.state_token,
                self.acc_token,
                self.causal,
            ],
            include=["../../../../../projects/galaxy/attention_local/flash_local.hpp"],
        )
        down = km.Broadcast(
            self,
            along="column",
            length=LENGTH,
            mover="A",
            window=BROADCAST_WINDOW,
        )
        jobs = tables.pages(self.rank, 1).entries(Job, first=8, size=8, count=0)
        for job in jobs:
            r0 = job.q // DT
            qs = km.Stream(
                q.tiles[
                    r0 : min(
                        r0 + R * job.stride, r0 + job.rows * job.stride
                    ) : job.stride,
                    0:DT,
                ],
                block=QB,
            )
            sink = km.Stream(
                sinks.tiles[
                    r0 : min(
                        r0 + R * job.stride, r0 + job.rows * job.stride
                    ) : job.stride,
                    0:1,
                ],
                block=R,
            )
            m0 = variant * MASKS
            mask = km.Stream(masks.tiles[m0 : m0 + MASKS, 0:1], block=MASKS)
            k0 = job.k // DT
            v0 = job.v // DV
            for b in range((job.keys + KB - 1) // KB):
                kr = k0 + b * KB
                vr = v0 + b * KB
                if down.sender():
                    block = down.cast(
                        kv.tiles[kr : min(kr + KB, k0 + job.keys), 0:DT].T,
                        kv.tiles[vr : min(vr + KB, v0 + job.keys), 0:DV],
                    )
                else:
                    block = down.arrive()
                # the entries' mask of the block's keys: block b >= 1 is entry tiles
                # (b - 1) KB on (block 0, the window's, reads tiles 0 on unused)
                e0 = (b - min(b, 1)) * KB
                entry = km.Stream(
                    entries.tiles[e0 : min(e0 + KB, e0 + job.keys - b * KB), 0:1],
                    block=KB,
                )
                km.cxx(
                    """
                    flash_local::cb_queries = qs_buffer;
                    flash_local::cb_sinks = sink_buffer;
                    flash_local::cb_masks = mask_buffer;
                    flash_local::cb_entries = entry_buffer;
                    using Flash = flash_local::Flash<
                        block_buffer, ones, p, state, acc, state_token, acc_token,
                        causal>;
                    Flash::block(
                        job_rows, job_start, job_keys, job_stride,
                        b * FLASH_BLOCK_KEYS,
                        b * (FLASH_BLOCK_KEYS / FLASH_SUB_CHUNK)
                    );
                    """,
                    takes=[
                        qs,
                        sink,
                        mask,
                        entry,
                        block,
                        job,
                        b,
                        self.ones,
                        self.p,
                        self.state,
                        self.acc,
                        self.state_token,
                        self.acc_token,
                        self.causal,
                    ],
                )
            o0 = job.o // DV
            out = km.Stream(
                o.tiles[
                    o0 : min(
                        o0 + R * job.stride, o0 + job.rows * job.stride
                    ) : job.stride,
                    0:DV,
                ],
                write=True,
                block=DV,
            )
            for row in range(job.rows):
                km.cxx(
                    """
                    flash_local::cb_output = out_buffer;
                    flash_local::Flash<
                        ones, ones, p, state, acc, state_token, acc_token, causal
                    >::finalize(row);
                    """,
                    takes=[
                        out,
                        row,
                        self.ones,
                        self.p,
                        self.state,
                        self.acc,
                        self.state_token,
                        self.acc_token,
                        self.causal,
                    ],
                )
        km.cxx(
            """
            cb_pop_front(ones, 1);
            #ifdef FLASH_CAUSAL
            cb_pop_front(causal, 1);
            #endif
            PACK((flash::restore_constants()));
            """,
            takes=[self.ones, self.causal],
        )


def dims(key_tiles):
    """The call: one head of 64 query tile rows (a tile row a query head) against
    `key_tiles` key tiles (the sink's first), D 512."""
    return pl.dims(1, 1, 1, HEADS * TILE, key_tiles * TILE, HEAD_DIM, HEAD_DIM, False)


def island_ranks(session):
    """The ranks of the session's first island, where the piece runs: all of a card's
    session, less the link cores on joined cards (75 of 79 on Wormhole)."""
    graph = getattr(session, "graph", None)
    return session.ranks if graph is None else graph.count(0)


def grid(session):
    """The class's broadcast rectangle: the whole rows of the first island's ranks (a
    Tensix dispatcher's row left out), columns x rows."""
    columns = session.nocs().grid[0]
    return columns, island_ranks(session) // columns


def group(session):
    """attend's group: the rectangle's ranks, fixed at load."""
    columns, rows = grid(session)
    return (0, columns * rows)


def plan_of(session, key_tiles):
    """The plan of `key_tiles` on the rectangle at the class's sizes; refused when it is
    not the class's."""
    sizes = {"rows": R, "block_keys": KB, "length": LENGTH}
    made = pl.make(dims(key_tiles), grid(session), **sizes)
    if key_tiles > KEY_TILES or (made.r, made.kb, made.length) != (R, KB, LENGTH):
        raise km.KomorebiError(f"plan {made.summary()}: the class is of {R, KB}")
    return made


def tables_tensor(session, key_tiles):
    """The job tables of `key_tiles` key tiles as their DRAM tensor."""
    return card.table_tensor(session, pl.tables(plan_of(session, key_tiles)))


def words(q, kv, o, tables, sinks, masks, variant, entries):
    """attend's arguments: one pass of every job, K and V the key list (the window,
    then the selected entries), the entries' mask."""
    return (q, kv, o, tables, sinks, masks, variant, entries)


def sink_values(sink):
    """A layer's sink tiles [64 x 32, 32]: column 0 sink_h / scale, the other columns
    MASKED (a query tile row h's tile)."""
    tiles = np.full((HEADS, TILE, TILE), MASKED, np.float32)
    tiles[:, :, 0] = np.asarray(sink, np.float32)[:, None] / np.float32(SCALE)
    return tiles.reshape(-1, TILE)


def decode_variant(pos):
    """The mask variant of a decode token at pos (its row 0)."""
    return min(pos, WINDOW - 1)


def prefill_variant(start):
    """The mask variant of a block of 32 at start (a multiple of 32)."""
    if start % TILE:
        raise ValueError(f"a prefill block starts at a multiple of {TILE}, not {start}")
    return DECODE_VARIANTS + start // TILE % RING_TILES + RING_TILES * (start >= WINDOW)


def entry_values(seen):
    """The entries' mask [16 x 32, 32] of seen [32, n <= 512] (query row, entry)."""
    full = np.zeros((TILE, ENTRY_TILES * TILE), bool)
    full[:, : seen.shape[1]] = seen
    tiles = full.reshape(TILE, ENTRY_TILES, TILE).transpose(1, 0, 2)
    return np.where(tiles, 0.0, MASKED).astype(np.float32).reshape(-1, TILE)


def visible(pos, tokens, prefill):
    """[32, 32 x 6] whether query row i sees key column j of the window (key tile 0
    the sink's, never; 1 .. 4 the ring's slots; 5 the block's own): a decode token at
    pos in row 0 (the ring holding its own key), or a block at pos's 32 rows."""
    seen = np.zeros((TILE, (1 + RING_TILES + 1) * TILE), bool)
    slots = np.arange(WINDOW)
    for i in range(tokens):
        if prefill:
            # slot j holds the last position = j (mod 128) before the block
            held = pos - WINDOW + (slots - pos) % WINDOW
            now = pos + i
            seen[i, TILE : TILE + WINDOW] = (held >= 0) & (held > now - WINDOW)
            seen[i, TILE + WINDOW : TILE + WINDOW + i + 1] = True
        else:
            held = pos - (pos - slots) % WINDOW  # the last position = j up to pos
            seen[i, TILE : TILE + WINDOW] = held >= 0
    return seen


def mask_values():
    """The masks' table [136 x 5 x 32, 32]: a variant's 5 tiles (key tiles 1 .. 5), 0
    where the row sees the key, MASKED where not; decode variants 0 .. 127 then the 8
    prefill variants (each of a start it stands for: 0, 32, 64, 96, 128, 160, ...)."""
    out = [visible(pos, TILE, False) for pos in range(DECODE_VARIANTS)]
    out += [visible(s, TILE, True) for s in (0, 32, 64, 96, 128, 160, 192, 224)]
    seen = np.stack(out)[:, :, TILE:]  # [variants, 32, 5 x 32]
    tiles = seen.reshape(len(out), TILE, MASKS, TILE).transpose(0, 2, 1, 3)
    return np.where(tiles, 0.0, MASKED).astype(np.float32).reshape(-1, TILE)


def reference(q, kv, sink, visible):
    """float64 attention of the layer: q [64, 32, 512] (head, token), kv [n, 512],
    visible [32, n] (token, key), the sink a logit with no value."""
    q, kv = np.asarray(q, np.float64), np.asarray(kv, np.float64)
    scores = np.einsum("htd,nd->htn", q, kv) * SCALE
    scores = np.where(visible[None], scores, -np.inf)
    sink = np.asarray(sink, np.float64)[:, None, None]
    m = np.maximum(scores.max(-1, keepdims=True), sink)
    p = np.exp(scores - m)
    total = p.sum(-1, keepdims=True) + np.exp(sink - m)
    return np.einsum("htn,nd->htd", p / total, kv)
