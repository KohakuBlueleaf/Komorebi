/* Decode: what the X280 thread, the operations' parts and Python share. Plain C. */
#pragma once

#include <stdint.h>

/* The operations, called after a product's sum at each group's root: add (a7 the
 * residual, a9 its rows' squares out), gate and scale (a7 squares in, a9 epsilon). */
enum { DC_OVERLAY_ADD = 1, DC_OVERLAY_GATE = 2, DC_OVERLAY_SCALE = 3 };

/* Buffers of the operations, beside the K split's (indices 0, 1, 2, 5, 6, 16, 24). */
enum {
    DC_CB_BLOCK = 3,    /* add: the residual's block, oh x ow tiles */
    DC_CB_Y = 4,        /* add: Y again in bf16, for its rows' squares */
    DC_CB_PARTS = 7,    /* scale: the row's sums of squares, a tile a group */
    DC_CB_CONSTANT = 8, /* add, scale: the identity tile and the mean tile */
    DC_CB_SQUARES = 9,  /* scale: the row's sums of squares on the diagonal */
    DC_CB_SCALE = 10,   /* scale: each row's 1 / rms in every column */
    DC_CB_UP = 11,  /* gate: up's half of the sum in bf16, as the reference rounds it */
    DC_CB_PART = 12 /* add: this block's sums of squares on the diagonal */
};

/* The row's tiles (the model's width); DC_PARTS the tiles of a row's sums of squares
 * (o's and down's groups: each root adds its block's). */
enum { DC_ROW_TILES = 64, DC_PARTS = 8 };

/* Every operation's a8: the identity and the mean tile. The norm's weight is in the
 * next W's rows; that product's root (scale, gate) multiplies the sum by 1 / rms. */
enum { DC_CONSTANT_IDENTITY = 0, DC_CONSTANT_MEAN = 1, DC_CONSTANTS = 2 };

/* LAYERS' a4 bits: 1 products only, 2 (with 1) parts only, 4 q, k, v apart, 8 (with 4)
 * q to o overlapped. The words: handler, plans, operation words, in order. */
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
    DC_WORD_EPSILON, /* the norm's epsilon, float bits */
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
    DC_LAYER_WORDS = 16
};

/* A layer's commands, in order: their kinds, for the stats. */
enum {
    DC_STEP_QKV,
    DC_STEP_Q,
    DC_STEP_K,
    DC_STEP_V,
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
    DC_STATS_SLOTS = 8
};
