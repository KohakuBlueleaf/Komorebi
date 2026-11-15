/* Decode: what the X280 thread, the operations' parts and Python share. Plain C. */
#pragma once

#include <stdint.h>

/* The operations, overlays of the K-split program, called after a product's sum at
 * the root of each group (add, gate) or as a command of their own (norm). */
enum { DC_OVERLAY_ADD = 1, DC_OVERLAY_GATE = 2, DC_OVERLAY_NORM = 3 };

/* Buffers of the operations, beside the K split's (indices 0, 1, 2, 5, 6, 16, 24). */
enum {
    DC_CB_BLOCK = 3,    /* add: the block of the other operand, oh x ow tiles */
    DC_CB_ROW = 4,      /* norm: x's whole row, DC_ROW_TILES tiles */
    DC_CB_WEIGHT = 7,   /* norm: the core's tiles of the norm's weight */
    DC_CB_CONSTANT = 8, /* norm: the identity tile and the mean tile */
    DC_CB_SQUARES = 9,  /* norm: the row's sums of squares on the diagonal */
    DC_CB_SCALE = 10,   /* norm: each row's 1 / rms in every column */
    DC_CB_NORMED = 17,  /* norm: the core's tiles of the output */
    DC_CB_UP = 11 /* gate: up's half of the sum in bf16, as the reference rounds it */
};

/* The norm's shape: the row's tiles (the model's width) and the cores of a command,
 * each DC_ROW_TILES / DC_NORM_CORES of the output. */
enum { DC_ROW_TILES = 64, DC_NORM_CORES = 16, DC_NORM_SHARE = 4 };

/* A norm command (SPLIT mode 2): a1 epsilon's float bits, a2 the output, a7 x, a8 the
 * weight (its row in every row of a tile row), a9 the constants (identity, mean). */
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
    DC_WORD_NORM,
    DC_WORD_EPSILON, /* the norm's epsilon, float bits */
    DC_WORDS
};

/* Region `tensors`: the activations' addresses. */
enum {
    DC_TENSOR_X,   /* the residual stream, [32, D] */
    DC_TENSOR_H,   /* the norm's output */
    DC_TENSOR_QKV, /* q, k, v side by side: [32, 4096]; apart, q only */
    DC_TENSOR_K,   /* apart: k and v */
    DC_TENSOR_V,
    DC_TENSOR_GATED, /* silu(h Wg) * (h Wu) */
    DC_TENSOR_SUMS,  /* bare: gate_up's sum, [32, 12288] */
    DC_TENSOR_CONSTANTS,
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
    DC_W_NORM1,
    DC_W_NORM2,
    DC_LAYER_WORDS = 16
};

/* A layer's commands, in order: their kinds, for the stats. */
enum {
    DC_STEP_NORM1,
    DC_STEP_QKV,
    DC_STEP_Q,
    DC_STEP_K,
    DC_STEP_V,
    DC_STEP_O,
    DC_STEP_NORM2,
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
