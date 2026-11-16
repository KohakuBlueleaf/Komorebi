"""The decode layer as one class: qkv, attend, o, gate_up and down as handlers, each
product the hand's K split with the hand's operation on the group's sum at its root
(the bodies call layer.hpp, the hand's C++ with the class's buffers), run on the card
beside the hand's layer (projects/decode) at T 256 (README.md)."""

import math

import numpy as np

import komorebi as km
from projects.decode import model as md

MODEL = md.MODELS["qwen3-1.7b"]
GROUPS = {"qkv": 7, "o": 15, "gate_up": 2, "down": 15}  # md.plans' on 120 cores
HIDDEN_TILES = MODEL.hidden // md.TILE
EPSILON = int(np.array([MODEL.eps], np.float32).view(np.uint32)[0])
CONTEXT = 256  # the cache's rows a KV head (T 256)
KEYS_MOST = math.ceil((CONTEXT // md.TILE - 1) / GROUPS["o"]) + 1  # card.keys_most

# qkv's root: the norm's 1 / rms, then on q's and k's blocks each head's norm and
# rotation, else the scaled sum.
SCALE = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const bool room = out != 0;  // pack's room for the write: guards its packs
if (
    !compute.wait_front(head_tiles, kItemPages) ||
    !layer::rms_scale<scale>(
        compute, parts_buffer, head_tiles, scale, scale, EPSILON, live
    )
) {
    return;
}
if (first < QK_TILES) {
    if (!layer::heads(
        compute, whole_buffer, scale, y, head_tiles, rope, scale, normed,
        out_buffer, WIDTH, EPSILON, live, room
    )) {
        return;
    }
} else {
    layer::scaled(
        compute, whole_buffer, scale, out_buffer, WIDTH, room, false
    );
    layer::restore(scale, out_buffer);
}
cb_pop_front(head_tiles, kItemPages);
"""

# qkv's mover A: the norm's constants; on q's and k's blocks the head's norm weight
# and the rotation's rows (scratch: `again`, which qkv does not use).
QK_READ = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const bool qk_block = first < QK_TILES;
const uint32_t key = first >= QK_TILES / 3 * 2 ? 1u : 0u;
if (
    !layer::read_item(movement, command, qk, qk_block, layer_index, key, head_tiles) ||
    (qk_block &&
     !layer::read_rotary(movement, command, qk, position, live, again, rope))
) {
    return false;
}
"""

# The constants' item (the identity, the mean tile) on mover A, for add and gate.
CONSTANTS_READ = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
if (!layer::read_item(movement, command, constants, false, 0, 0, head_tiles)) {
    return false;
}
"""

# o's and down's root: Y = the sum + the residual, then Y's rows' squares.
ADD = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const bool room = out != 0;  // pack's room for the write: guards its packs
if (
    !compute.wait_front(head_tiles, kItemPages) ||
    !layer::residual_add<y>(
        compute, whole_buffer, residual_buffer, head_tiles, out_buffer,
        part_buffer, y, WIDTH, room
    )
) {
    return;
}
cb_pop_front(head_tiles, kItemPages);
"""

# gate_up's root: the norm's 1 / rms, then silu(gate) * up.
GATE = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const bool room = out != 0;  // pack's room for the write: guards its packs
if (
    !compute.wait_front(head_tiles, kItemPages) ||
    !layer::rms_scale<scale>(
        compute, parts_buffer, head_tiles, scale, scale, EPSILON, live
    ) ||
    !layer::gate(
        compute, whole_buffer, scale, y, out_buffer, WIDTH, live, room
    )
) {
    return;
}
cb_pop_front(head_tiles, kItemPages);
"""

# attend, mover A: the constants, this core's cached keys, the root's token key (pages
# 0-3 of `again`, compute's copy) appended to the cache; each mover takes its half.
ATTEND_KEYS = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const Attend at = attend_of(rank, O_GROUP, position, live);
if (
    !layer::read_attend_constants(movement, command, table, at, head_tiles) ||
    !layer::read_cache(movement, command, at, head, rope, keys) ||
    (rank == 0 && (
        !layer::wait(movement, again, DC_HEAD_TILES) ||
        !layer::append(
            movement, command, at, head, rope, keys, get_read_ptr(again), again
        )
    ))
) {
    return false;
}
if (rank == 0) {
    layer::taken_half(again, DC_HEAD_TILES);
}
layer::pad(at, rope);
"""

# attend, mover B: this core's cached values, the root's token value (pages 4-7 of
# `again`) appended.
ATTEND_VALUES = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const Attend at = attend_of(rank, O_GROUP, position, live);
if (
    !layer::read_cache(movement, command, at, head, cache_values, values) ||
    (rank == 0 && (
        !layer::wait(movement, again, DC_HEAD_TILES) ||
        !layer::append(
            movement, command, at, head, cache_values, values,
            get_read_ptr(again) + DC_HEAD_TILES * kLayerTileBytes, again
        )
    ))
) {
    return false;
}
if (rank == 0) {
    layer::taken_half(again, DC_HEAD_TILES);
}
layer::pad(at, cache_values);
"""

# attend's root, compute: the token's key and value tiles of qkv into `again`.
TOKEN = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
layer::token_copy<scale>(compute, token_key_buffer, token_value_buffer, HEAD, again);
"""

# attend, compute: the scores, each query head's state (maximum, sum, P V).
ATTEND = """
#include "../../../../../experiments/language/decode_class/layer.hpp"
const Attend at = attend_of(rank, O_GROUP, position, live);
if (!layer::attend_state<scale>(
    compute, at, qs_buffer, rope, cache_values, head_tiles, y, scale, rescale,
    normed, state, scale
)) {
    return;
}
"""

# The merge of an arrival into the state (in place), as the hand's tree merges it.
MERGE = """
if (!layer::merge(
    compute, arrival_buffer, arrival_index, state, kFieldRescale, kFieldHeadTiles,
    kFieldScale, live
)) {
    return;
}
"""

# A sender's last state token, taken (its state goes on from mover B).
SENT = """
if (rank != 0 && !layer::state_taken(compute, scale)) {
    return;
}
"""

# The root: each output tile over its head's sum, bf16, into a's block.
FINISH = """
const bool room = out != 0;  // pack's room for the write: guards its packs
if (!layer::finish(compute, state, scale, out_buffer, live, room)) {
    return;
}
layer::restore(scale, out_buffer);
"""


class DecodeLayer(
    km.Program,
    defines={  # the header's own sizes (layer.hpp)
        "LAYER_CACHE_TILES": CONTEXT // md.TILE,
        "LAYER_KEYS_MOST": KEYS_MOST,
    },
):
    """One decode layer: each product Y = A W by the hand's K split (groups of S
    ranks, a group WIDTH columns, a rank its km::partition of K in CHUNK-tile chunks,
    summed up a tree), the root running the hand's operation on the sum."""

    WIDTH = km.constant(md.BLOCK_WIDTH)  # every plan's 1 x 8 block
    CHUNK = km.constant(md.CHUNK)
    W_BLOCK = CHUNK * WIDTH
    HALF = km.constant(md.GATE_TILES)  # gate_up's block: the gate's 4, then up's 4
    PARTS = km.constant(8)  # a row's squares: a tile a group of o's and down's
    QKV_GROUP = km.constant(GROUPS["qkv"])
    O_GROUP = km.constant(GROUPS["o"])
    GATE_UP_GROUP = km.constant(GROUPS["gate_up"])
    DOWN_GROUP = km.constant(GROUPS["down"])
    QKV_K = km.constant(HIDDEN_TILES)
    O_K = km.constant(MODEL.heads * MODEL.head_dim // md.TILE)
    GATE_UP_K = km.constant(HIDDEN_TILES)
    DOWN_K = km.constant(MODEL.mlp // md.TILE)
    EPSILON = km.constant(EPSILON)  # the norms' epsilon, float bits
    Q_TILES = km.constant(MODEL.heads * MODEL.head_dim // md.TILE)  # qkv's q, k, v
    KV_TILES = km.constant(MODEL.kv_heads * MODEL.head_dim // md.TILE)
    QK_TILES = Q_TILES + KV_TILES  # the blocks whose heads are normed and rotated
    HEAD = km.constant(MODEL.head_dim // md.TILE)  # a head's tiles
    sums: km.L1[km.fp32, 1, WIDTH, 32, 32]
    # attend's root: the token's key and value, from pack to both movers (each takes
    # its half); qkv's mover A: scratch for the rotation's rows
    again: km.L1[km.bf16, 1, WIDTH, 32, 32] = km.own()
    # qkv: the scaled q or k block; add: Y again; gate: up's half in bf16
    y: km.L1[km.bf16, 1, WIDTH, 32, 32] = km.own()
    # one tile, pack to unpack, one live at a time: the norms' squares then 1 / rms;
    # attend's row maximum, then the state's token
    scale: km.L1[km.bf16, 1, 32, 32] = km.own()
    normed: km.L1[km.bf16, 4, 32, 32] = km.own()  # a head's 4 tiles normed
    head_tiles: km.L1[km.bf16, 8, 32, 32] = km.own(push="mover_a")
    rope: km.L1[km.bf16, 8, 32, 32] = km.own(push="mover_a")  # cos's 4, sin's 4
    # attend's, beside the own buffers above that it takes in turn: head_tiles its
    # constants, rope its keys, y its scores, normed P
    cache_values: km.L1[km.bf16, 8, 32, 32] = km.own(push="mover_b")
    # the maximum in every column (a page of two), then the merge's two factors
    rescale: km.L1[km.fp32, 2, 32, 32] = km.own()
    state: km.L1[km.fp32, 12, 32, 32] = km.own(push=None)  # two heads' max, sum, PV

    @km.tile_function(fidelity="HiFi2")
    def project(x, w):
        return x @ w

    @km.handler
    def qkv(
        self,
        x: km.Tensor,
        w_qkv: km.Tensor,
        qkv: km.Tensor,
        squares: km.Tensor,
        qk: km.Tensor,
        pattern: km.Pattern,
        position: int,
        layer_index: int,
        live: int,
    ):
        share = self.rank % self.QKV_GROUP  # the rank in its group: km::partition's
        first = self.rank // self.QKV_GROUP * self.WIDTH
        k0 = share * (self.QKV_K // self.QKV_GROUP) + min(
            share, self.QKV_K % self.QKV_GROUP
        )
        k1 = (share + 1) * (self.QKV_K // self.QKV_GROUP) + min(
            share + 1, self.QKV_K % self.QKV_GROUP
        )
        for ab, wb in zip(
            km.Stream(x.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_qkv.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.QKV_GROUP)
        if self.rank == self.leader:
            parts = km.Stream(squares.pages(0, self.PARTS), block=self.PARTS)
            out = km.Stream(
                qkv.tiles[0:1, first : first + self.WIDTH],
                write=True,
                block=self.WIDTH,
            )
            km.cxx(
                QK_READ,
                takes=[qk, first, layer_index, position, live, self.QK_TILES,
                       self.head_tiles,
                       self.rope, self.again],
                reads=[qk],
                on="mover_a",
            )  # fmt: skip
            km.cxx(
                SCALE,
                takes=[whole, parts, out, first, live, self.WIDTH, self.EPSILON,
                       self.QK_TILES, self.scale,
                       self.y, self.head_tiles, self.rope, self.normed],
            )  # fmt: skip

    @km.handler
    def attend(
        self,
        qkv: km.Tensor,
        keys: km.Tensor,
        values: km.Tensor,
        a: km.Tensor,
        table: km.Tensor,
        position: int,
        live: int,
    ):
        rank = self.rank % self.O_GROUP  # o's plan: a group a KV head
        head = self.rank // self.O_GROUP
        first = head * self.WIDTH  # its two query heads' tiles of qkv and of a
        qs = km.Stream(qkv.tiles[0:1, first : first + self.WIDTH], block=self.WIDTH)
        # each group's root (self.leader is rank 0 until the reduce below): the
        # token's key and value, for the append
        if self.rank % self.O_GROUP == 0:
            # blocks of qkv's shapes, so they share its read buffers (no index is
            # left): the key at tile 0 of a W_BLOCK, the value at HEAD of a WIDTH
            key = self.Q_TILES + head * self.HEAD
            value = key + self.KV_TILES - self.HEAD  # base + step x head: a chain's
            token_key = km.Stream(
                qkv.tiles[0:1, key : key + self.W_BLOCK], block=self.W_BLOCK
            )
            token_value = km.Stream(
                qkv.tiles[0:1, value : value + self.WIDTH], block=self.WIDTH
            )
            km.cxx(
                TOKEN, takes=[token_key, token_value, self.HEAD, self.scale, self.again]
            )
        km.cxx(
            ATTEND_KEYS,
            takes=[keys, table, rank, head, position, live, self.O_GROUP,
                   self.head_tiles, self.rope, self.again],
            reads=[table],
            writes=[keys],
            on="mover_a",
        )  # fmt: skip
        km.cxx(
            ATTEND_VALUES,
            takes=[values, rank, head, position, live, self.O_GROUP,
                   self.cache_values, self.again],
            writes=[values],
            on="mover_b",
        )  # fmt: skip
        km.cxx(
            ATTEND,
            takes=[qs, rank, position, live, self.O_GROUP, self.rope,
                   self.cache_values,
                   self.head_tiles, self.y, self.rescale, self.normed,
                   self.state, self.scale],
        )  # fmt: skip
        km.reduce(self.state, form="tree", group=self.O_GROUP, combine=MERGE)
        km.cxx(SENT, takes=[rank, self.scale])
        if self.rank == self.leader:
            out = km.Stream(
                a.tiles[0:1, first : first + self.WIDTH], write=True, block=self.WIDTH
            )
            km.cxx(FINISH, takes=[out, live, self.state, self.scale])

    @km.handler
    def o(
        self,
        attended: km.Tensor,
        w_o: km.Tensor,
        x1: km.Tensor,
        x: km.Tensor,
        squares: km.Tensor,
        constants: km.Tensor,
        pattern: km.Pattern,
    ):
        share = self.rank % self.O_GROUP
        group = self.rank // self.O_GROUP
        first = group * self.WIDTH
        k0 = share * (self.O_K // self.O_GROUP) + min(share, self.O_K % self.O_GROUP)
        k1 = (share + 1) * (self.O_K // self.O_GROUP) + min(
            share + 1, self.O_K % self.O_GROUP
        )
        for ab, wb in zip(
            km.Stream(attended.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_o.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.O_GROUP)
        if self.rank == self.leader:
            out = km.Stream(
                x1.tiles[0:1, first : first + self.WIDTH],
                write=True,
                block=self.WIDTH,
            )
            part = km.Stream(squares.tiles[0:1, group : group + 1], write=True)
            residual = km.Stream(
                x.tiles[0:1, first : first + self.WIDTH], block=self.WIDTH
            )
            km.cxx(
                CONSTANTS_READ,
                takes=[constants, self.head_tiles],
                reads=[constants],
                on="mover_a",
            )
            km.cxx(
                ADD,
                takes=[whole, residual, out, part, self.WIDTH, self.y,
                       self.head_tiles],
            )  # fmt: skip

    @km.handler
    def gate_up(
        self,
        h: km.Tensor,
        w_gate_up: km.Tensor,
        gated: km.Tensor,
        squares: km.Tensor,
        constants: km.Tensor,
        pattern: km.Pattern,
        live: int,
    ):
        share = self.rank % self.GATE_UP_GROUP
        first = self.rank // self.GATE_UP_GROUP * self.WIDTH
        k0 = share * (self.GATE_UP_K // self.GATE_UP_GROUP) + min(
            share, self.GATE_UP_K % self.GATE_UP_GROUP
        )
        k1 = (share + 1) * (self.GATE_UP_K // self.GATE_UP_GROUP) + min(
            share + 1, self.GATE_UP_K % self.GATE_UP_GROUP
        )
        for ab, wb in zip(
            km.Stream(h.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_gate_up.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.GATE_UP_GROUP)
        if self.rank == self.leader:
            parts = km.Stream(squares.pages(0, self.PARTS), block=self.PARTS)
            out = km.Stream(
                gated.tiles[0:1, first // 2 : first // 2 + self.HALF],
                write=True,
                block=self.HALF,
            )
            km.cxx(
                GATE,
                takes=[whole, parts, out, live, self.WIDTH, self.EPSILON,
                       self.scale,
                       self.y, self.head_tiles],
            )  # fmt: skip
            km.cxx(
                CONSTANTS_READ,
                takes=[constants, self.head_tiles],
                reads=[constants],
                on="mover_a",
            )

    @km.handler
    def down(
        self,
        g: km.Tensor,
        w_down: km.Tensor,
        x: km.Tensor,
        x1: km.Tensor,
        squares: km.Tensor,
        constants: km.Tensor,
        pattern: km.Pattern,
    ):
        share = self.rank % self.DOWN_GROUP
        group = self.rank // self.DOWN_GROUP
        first = group * self.WIDTH
        k0 = share * (self.DOWN_K // self.DOWN_GROUP) + min(
            share, self.DOWN_K % self.DOWN_GROUP
        )
        k1 = (share + 1) * (self.DOWN_K // self.DOWN_GROUP) + min(
            share + 1, self.DOWN_K % self.DOWN_GROUP
        )
        for ab, wb in zip(
            km.Stream(g.tiles[0:1, k0:k1], block=self.CHUNK, pattern=pattern),
            km.Stream(
                w_down.tiles[k0:k1, first : first + self.WIDTH],
                block=self.W_BLOCK,
                pattern=pattern,
            ),
        ):
            self.sums += self.project(ab, wb)
        whole = km.reduce(self.sums, form="tree", group=self.DOWN_GROUP)
        if self.rank == self.leader:
            out = km.Stream(
                x.tiles[0:1, first : first + self.WIDTH], write=True, block=self.WIDTH
            )
            part = km.Stream(squares.tiles[0:1, group : group + 1], write=True)
            residual = km.Stream(
                x1.tiles[0:1, first : first + self.WIDTH], block=self.WIDTH
            )
            km.cxx(
                CONSTANTS_READ,
                takes=[constants, self.head_tiles],
                reads=[constants],
                on="mover_a",
            )
            km.cxx(
                ADD,
                takes=[whole, residual, out, part, self.WIDTH, self.y,
                       self.head_tiles],
            )  # fmt: skip


PLANS = md.plans(MODEL, halves=(md.DEPTH, 1))  # the hand's, a command a step
QKV_CORES, O_CORES, GATE_UP_CORES, DOWN_CORES = (
    PLANS[md.plan_of(MODEL, name)].cores() for name in GROUPS
)


class DecodeLayerChain(DecodeLayer):
    """The five handlers as overlays (the resident kernels small) with the chain's
    walk: the layer as one chain command (run_layer.py --chain)."""

    qkv = km.handler(overlay=True)(DecodeLayer.qkv.function)
    attend = km.handler(overlay=True)(DecodeLayer.attend.function)
    o = km.handler(overlay=True)(DecodeLayer.o.function)
    gate_up = km.handler(overlay=True)(DecodeLayer.gate_up.function)
    down = km.handler(overlay=True)(DecodeLayer.down.function)

    @km.chain
    def layer(
        self,
        x,
        w_qkv,
        qkv,
        squares1,
        qk,
        keys,
        values,
        a,
        table,
        w_o,
        x1,
        squares2,
        w_gate_up,
        gated,
        w_down,
        constants,
        position: int,
    ):
        """The layer's five calls, layer 0 at `position`: x1, written by o, is down's
        residual two calls on (km.past)."""
        self.qkv(
            x, w_qkv, qkv, squares1, qk, "AB", position, 0, 1, cores=(0, QKV_CORES)
        )
        self.attend(qkv, keys, values, a, table, position, 1, cores=(0, O_CORES))
        self.o(a, w_o, x1, x, squares2, constants, "AB", cores=(0, O_CORES))
        self.gate_up(
            x1,
            w_gate_up,
            gated,
            squares2,
            constants,
            "AAAB",
            1,
            cores=(0, GATE_UP_CORES),
        )
        self.down(
            gated,
            w_down,
            x,
            km.past(x1),
            squares1,
            constants,
            "AB",
            cores=(0, DOWN_CORES),
        )


class DecodeLayerMixed(DecodeLayerChain):
    """o and down resident (their tile code in the kernels), qkv, attend and gate_up
    overlays: the chain with resident and overlay handlers (run_layer.py --mixed)."""

    o = km.handler(DecodeLayer.o.function)
    down = km.handler(DecodeLayer.down.function)
