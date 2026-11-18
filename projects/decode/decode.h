/* Decode: what the X280 thread, the operations' parts and Python share. Plain C. */
#pragma once

#include <stdint.h>

/* At a group's root after the sum: add (a7 residual, a9 squares out), gate (a7 squares
 * in), scale (a7), largest (scale, then the block's largest); attend on every core. */
enum {
    DC_OVERLAY_ADD = 1,
    DC_OVERLAY_GATE = 2,
    DC_OVERLAY_SCALE = 3,
    DC_OVERLAY_ATTEND = 4,
    DC_OVERLAY_LARGEST = 5
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
    /* attend: the head's query tiles, the ones, row-0 and identity tiles, the root's
     * two masks; keys streamed, values, scores, P (bf16) held for the command */
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
    DC_MASKS = 63,       /* mask k: row i's columns past i + k - 31 at DC_MASKED */
    DC_MASK_ZERO = 31,   /* the mask of offset 0 */
    DC_ATTEND_ONES = 63, /* the constants' tile of ones */
    DC_ATTEND_ROW = 64,  /* the tile with ones in row 0 only */
    DC_ATTEND_IDENTITY = 65,
    DC_ATTEND_CONSTANTS = 66,
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

/* GENERATE: LAYERS' a0 to a6 with the head, a7 the prompt's ids (region `prompt`), a8
 * the end; `made`: (count, start), then (id, ticks, status) a token; result: count. */
enum {
    DC_HANDLER_GENERATE = 2,
    DC_MADE_WORDS = 4,
    DC_MADE_MOST = 1024,
    DC_PROMPT_MOST = 8192
};

/* LAYERS' a4 128: the layer as one command (the chain); 256 its first two only; 512 a
 * token as one (its records written once a form); 1024 kChainAheadOff (chain.h). */
enum {
    DC_LAYERS_CHAIN = 128,
    DC_LAYERS_PAIR = 256,
    DC_LAYERS_TOKEN = 512,
    DC_LAYERS_AHEAD_OFF = 1024
};

/* LAYERS' a4 bits 12 to 16: the records (qkv, attend, o, gate_up, down) whose input is
 * sent (kChainDataMark, never attend's); 2048: its DRAM copy acked too (chain.h). */
enum {
    DC_LAYERS_DATA_SHIFT = 12,
    DC_LAYERS_DATA_MASK = 0x1F,
    DC_LAYERS_DATA_DRAM = 2048
};

/* LAYERS' a4 bit 17: with a token as one command, the final norm's record (x into xn)
 * and the head's parts after its last layer's (each block's largest into LARGEST). */
enum { DC_LAYERS_HEAD = 0x20000 };

/* A token's last record's stamp row (the others: their step's, 0 to 4). */
enum { DC_CHAIN_LAST_ROW = 5 };

/* Every command's a3 bits 16 to 21: the rows that hold a token (1 to 32; 0 is 32), the
 * first ones; LAYERS' a6 the same for each step (positions a5 + step x rows on). */
enum { DC_ROWS_SHIFT = 16, DC_ROWS_MASK = 0x3F };

/* qkv's a3 with DC_QK: on q's and k's blocks the model's norm of each head and the
 * rotary embedding after scale; a8 the qk tensor (tiles), a9 position | layer << 16. */
enum {
    DC_QK = 0x400000,     /* a3 bit 22 */
    DC_CB_ROPE_ROWS = 26, /* DC_CB_TOKEN's: the reader's scratch, the band's rows */
    DC_CB_ROPE = 13,     /* DC_CB_QUERY's: those rows as tiles, cos's 4, then sin's 4 */
    DC_CB_HEAD = 3,      /* DC_CB_BLOCK's: the head's mean tile, the norm weight's 4 */
    DC_CB_NORMED = 11,   /* DC_CB_UP's: a head's 4 tiles normed, to be rotated */
    DC_HEAD_ITEM = 8,    /* DC_CB_HEAD's item: the buffer whole, 5 pages used */
    DC_QK_HEAD_MEAN = 2, /* the qk tensor's tiles: the identity, the mean, the head's */
    DC_QK_WEIGHTS = 3,   /* mean; layer l's q norm weight at 3 + 8 l, its k's at 7 + 8 l
                          */
    DC_QK_LAYERS = 64,   /* layers at most */
    DC_QK_ROTARY = 515,  /* 3 + 8 x 64: position p's row in page 515 + p / 4 */
    DC_ROPE_ROW = 512    /* a position's row: cos, then sin signed as rotate_half's */
};

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
    DC_WORD_EPSILON,      /* the norm's epsilon, float bits */
    DC_WORD_PLAN_ATTEND,  /* o's plan | the cores of every KV head's group << 8 */
    DC_WORD_ATTEND,       /* a6 of ATTEND */
    DC_WORD_CACHE_TILES,  /* the cache's key tiles a head (its rows / 32) */
    DC_WORD_GATE_TILES,   /* the chain: gated tiles a gate_up group writes */
    DC_WORD_ATTEND_NEEDS, /* the chain: attend's three ranges of qkv (chain.h) */
    DC_WORD_PLAN_HEAD = DC_WORD_ATTEND_NEEDS + 3, /* a head part's plan word */
    DC_WORD_LARGEST,                              /* a6 of the head's records */
    DC_WORD_HEAD,         /* the head's parts on the card (0: none) */
    DC_WORD_HEAD_COLUMNS, /* a part's columns */
    DC_WORD_VOCAB,        /* the head's columns that are tokens */
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
    DC_TENSOR_SQUARES1,  /* x's sums of squares: [32, DC_PARTS x 32] */
    DC_TENSOR_SQUARES2,  /* x1's */
    DC_TENSOR_A,         /* attention's output: [32, heads x 128] */
    DC_TENSOR_ATTEND,    /* attend's constants: DC_ATTEND_CONSTANTS tiles */
    DC_TENSOR_QK,        /* qkv's q and k norms and rotation (DC_QK_*), rows of 256 */
    DC_TENSOR_LARGEST,   /* the head's: a page a block, its largest and its column */
    DC_TENSOR_XN,        /* x's final norm, the head's input */
    DC_TENSOR_NORM,      /* the final norm's W: its weight on the diagonal */
    DC_TENSOR_HEAD = 16, /* the head's W, a part each, DC_HEAD_PARTS_MOST at most */
    DC_HEAD_PARTS_MOST = 8,
    DC_TENSOR_EMBED = 24, /* the embedding's own bf16 copy, a row a page (0: none) */
    DC_TENSORS = 32
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
    DC_STEP_CHAIN, /* the layer as one command (LAYERS' a4 128) */
    DC_STEPS
};

/* The chain thread's parts of a layer: its records written and flushed, the submit,
 * the wait (the command's event within it); GENERATE's embedding and choice. */
enum {
    DC_THREAD_RECORDS,
    DC_THREAD_SUBMIT,
    DC_THREAD_WAIT,
    DC_THREAD_EMBED,
    DC_THREAD_CHOOSE,
    DC_THREAD_GATHER, /* of the embedding's: its fetches */
    DC_THREAD_FETCH,  /* of the choice's: its fetch */
    DC_THREADS
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
    DC_ST_THREAD = 50, /* the chain's thread: DC_THREADS sums (low, high) of ticks */
    DC_STATS_WORDS = 64,
    DC_STATS_SLOTS = 64 /* a job each: the checks and 2 x 6 cases at 3 lengths */
};
