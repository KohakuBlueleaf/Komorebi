/* Decode: what the X280 thread, the operations' parts and Python share. Plain C. */
#pragma once

#include <stdint.h>

/* The operations at a group's root after the sum: add (a7 residual, a9 squares out),
 * gate, scale (a7 squares in, a9 epsilon); attend on every core of a group (below). */
enum {
    DC_OVERLAY_ADD = 1,
    DC_OVERLAY_GATE = 2,
    DC_OVERLAY_SCALE = 3,
    DC_OVERLAY_ATTEND = 4
};

/* Buffers of the operations, beside the K split's (indices 0, 1, 2, 5, 6, 16, 24). */
enum {
    DC_CB_BLOCK = 3,    /* add: the residual's block, oh x ow tiles */
    DC_CB_Y = 4,        /* add: Y again in bf16, for its rows' squares */
    DC_CB_PARTS = 7,    /* scale: the row's sums of squares, a tile a group */
    DC_CB_CONSTANT = 8, /* add, scale: the identity tile and the mean tile */
    DC_CB_SQUARES = 9,  /* scale: the row's sums of squares on the diagonal */
    DC_CB_SCALE = 10,   /* scale: each row's 1 / rms in every column */
    DC_CB_UP = 11, /* gate: up's half of the sum in bf16, as the reference rounds it */
    DC_CB_PART = 12, /* add: this block's sums of squares on the diagonal */
    /* attend: the head's two query tiles rows, the ones and the row-0 tile, the last
     * key tile's mask; keys streamed, values, scores, P (bf16) held for the command */
    DC_CB_QUERY = 13,
    DC_CB_ATTEND_CONSTANT = 14,
    DC_CB_MASK = 15,
    DC_CB_KEYS = 17,
    DC_CB_VALUES = 18,
    DC_CB_SCORES = 19,
    DC_CB_PROBABILITIES = 20,
    DC_CB_ROW_MAX = 21, /* a row's largest score in column 0 (bf16) */
    /* fp32, unpacked to DEST: the maximum in every column, the state (two items of a
     * maximum, a sum and the output's tiles a head), the merge's two rescales */
    DC_CB_REFERENCE = 22,
    DC_CB_STATE = 23,
    DC_CB_RESCALE = 25,
    DC_CB_TOKEN = 26, /* reader and writer: the token's key (value) tiles, to append */
    DC_CB_OUTBOX = 27 /* fp32: a core's last state, for the writer to send */
};

/* The row's tiles (the model's width); DC_PARTS the tiles of a row's sums of squares
 * (o's and down's groups: each root adds its block's). */
enum { DC_ROW_TILES = 64, DC_PARTS = 8 };

/* ATTEND, SPLIT's operation mode on o's plan, a group a KV head: a0 qkv, a1 keys, a2 a,
 * a3 form | DC_ATTEND_*, a7 values, a8 constants, a9 position | head's tiles << 16. */
enum {
    DC_HEAD_TILES = 4,   /* a head's width, 128 */
    DC_HEAD_GROUP = 2,   /* query heads a KV head */
    DC_STATE_TILES = 6,  /* a head's state: maximum, sum, 4 output tiles */
    DC_STATE_ITEM = 12,  /* both query heads' states: one message */
    DC_MASKS = 32,       /* mask r: columns above r at DC_MASKED */
    DC_ATTEND_ONES = 32, /* the constants' tile of ones */
    DC_ATTEND_ROW = 33,  /* the tile with ones in row 0 only */
    DC_ATTEND_IDENTITY = 34,
    DC_ATTEND_CONSTANTS = 35,
    DC_ATTEND_READ_ONLY = 0x200, /* a3: the reads, the append; no math, no merge */
    DC_ATTEND_LOCAL = 0x400,     /* a3: no merge (the root ends on its own part) */
    DC_ATTEND_EXP_TWICE = 0x800, /* a3: P's exponent taken twice (its cost, timed) */
    DC_KEY_BATCH = 4             /* key tiles a read batch */
};

#define DC_SCALE_BITS 0x3DB504F3u /* 1 / sqrt(128) */
#define DC_MASKED (-1e30f)

/* Every operation's a8: the identity and the mean tile. The norm's weight is in the
 * next W's rows; that product's root (scale, gate) multiplies the sum by 1 / rms. */
enum { DC_CONSTANT_IDENTITY = 0, DC_CONSTANT_MEAN = 1, DC_CONSTANTS = 2 };

/* LAYERS' a4: 1 products only, 2 parts only, 4 q k v apart (no attention), 8 overlap,
 * 16 attend's reads only, 32 no merge, 64 exponent twice; a5 the first position. */
enum { DC_HANDLER_LAYERS = 1 };

enum {
    DC_WORD_SPLIT,        /* the SPLIT handler */
    DC_WORD_REDUCE,       /* the reduce form */
    DC_WORD_EXCLUSIVE,    /* the command flag of a product */
    DC_WORD_PLAN_QKV,     /* plans: the index | its cores << 8; 2048 x 4096 */
    DC_WORD_PLAN_Q,       /* 2048 x 2048: q and o */
    DC_WORD_PLAN_KV,      /* 2048 x 1024 */
    DC_WORD_PLAN_GATE_UP, /* 2048 x 12288 */
    DC_WORD_PLAN_DOWN,    /* 6144 x 2048 */
    DC_WORD_ADD,          /* a6 of each operation: its overlay and parts */
    DC_WORD_GATE,
    DC_WORD_SCALE,
    DC_WORD_EPSILON,     /* the norm's epsilon, float bits */
    DC_WORD_PLAN_ATTEND, /* o's plan | the cores of every KV head's group << 8 */
    DC_WORD_ATTEND,      /* a6 of ATTEND */
    DC_WORD_CACHE_TILES, /* the cache's key tiles a head (its rows / 32) */
    DC_WORDS
};

/* Region `tensors`: the activations' addresses. */
enum {
    DC_TENSOR_X,   /* the residual stream, [32, D] */
    DC_TENSOR_X1,  /* the residual after attention (o's add) */
    DC_TENSOR_QKV, /* q, k, v side by side: [32, 4096]; apart, q only */
    DC_TENSOR_K,   /* apart: k and v */
    DC_TENSOR_V,
    DC_TENSOR_GATED, /* silu(h Wg) * (h Wu) */
    DC_TENSOR_SUMS,  /* bare: gate_up's sum, [32, 12288] */
    DC_TENSOR_CONSTANTS,
    DC_TENSOR_SQUARES1, /* x's sums of squares: [32, DC_PARTS x 32] */
    DC_TENSOR_SQUARES2, /* x1's */
    DC_TENSOR_A,        /* attention's output: [32, heads x 128] */
    DC_TENSOR_ATTEND,   /* attend's constants: DC_ATTEND_CONSTANTS tiles */
    DC_TENSORS = 16
};

/* Region `weights`: a layer's addresses, DC_LAYER_WORDS a layer. */
enum {
    DC_W_QKV,
    DC_W_Q,
    DC_W_K,
    DC_W_V,
    DC_W_O,
    DC_W_GATE_UP,
    DC_W_DOWN,
    DC_W_KEYS, /* the layer's cache: [KV heads x rows, 128] each */
    DC_W_VALUES,
    DC_LAYER_WORDS = 16
};

/* A layer's commands, in order: their kinds, for the stats. */
enum {
    DC_STEP_QKV,
    DC_STEP_Q,
    DC_STEP_K,
    DC_STEP_V,
    DC_STEP_ATTEND,
    DC_STEP_O,
    DC_STEP_GATE_UP,
    DC_STEP_DOWN,
    DC_STEPS
};

/* Region `stats`, DC_STATS_WORDS a job: status, done, ticks, then per step the ticks
 * summed, the most, the least; the overlapped span. 64-bit counts: low, high. */
enum {
    DC_ST_STATUS,
    DC_ST_DONE,
    DC_ST_TICKS_LOW,
    DC_ST_TICKS_HIGH,
    DC_ST_STEPS,
    DC_ST_STEP_WORDS = 4,
    DC_ST_SPAN_LOW = 48, /* overlapped: q's submit to o's event, summed (low, high) */
    DC_ST_SPAN_HIGH = 49,
    DC_STATS_WORDS = 64,
    DC_STATS_SLOTS = 32
};
