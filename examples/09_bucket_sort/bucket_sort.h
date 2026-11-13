/* Bucket sort: what the kernels (C++) and the thread (C) share. The host script
 * follows the same words; README.md explains every one. */
#pragma once

#include <stdint.h>

/* Step 1, COUNT: every core counts its partition of the keys per bucket and writes its
 * row of the counts table. Step 3, PLACE: every core writes each key to its place. */
#define BS_HANDLER_COUNT 1u
#define BS_HANDLER_PLACE 2u

/* The words of both commands: a0 keys, a1 N, a2 log2(B) | blocks << 8, a3 the blocks'
 * banks (8 bits each), a4 to a6 the blocks' tables, a7 the output (README.md). */

/* A bucket is the key's top log2(B) bits; buckets go in chunks of 16, one 64-byte line
 * of a table row, so that no cache line or NoC transfer holds two blocks' buckets. */
#define BS_CHUNK 16u
#define BS_BLOCKS_MAX 3u       /* one block of buckets per work cluster taking part */
#define BS_BUCKETS_MAX 16384u  /* a core keeps B counters in L1 */
#define BS_KEYS_PER_PAGE 1024u /* the keys and output tensors: 4 KiB pages */
#define BS_SLOTS 64u           /* PLACE: 16-byte staging slots for the 4-byte writes */

/* The thread's words, in this order (the host script writes them). */
enum {
    BS_W_KEYS_N,     /* N */
    BS_W_LOG2_B,     /* log2(B), 4 to 14 */
    BS_W_CORES,      /* the Tensix group: cores 0 to cores - 1, one table row each */
    BS_W_KEYS,       /* the keys tensor's DRAM address */
    BS_W_OUT,        /* the output tensor's DRAM address */
    BS_W_X280_CORES, /* 1, 4 or 12 */
    BS_W_REPEATS,    /* rounds of count, plan, place (the best time is kept) */
    BS_W_BANKS,      /* the DRAM bank of work cluster c in bits 8 (c - 1) */
    BS_WORDS
};

/* The thread's result words: the best time of each step in timer ticks (50 per
 * microsecond), the keys the plan placed (must be N), the rounds done. */
enum {
    BS_O_COUNT,
    BS_O_PLAN,
    BS_O_PLACE,
    BS_O_WHOLE,
    BS_O_PLACED,
    BS_O_ROUNDS,
    BS_OUT_WORDS
};

/* Block k of `blocks`: chunks [lo, hi) of `chunks`. The kernels and the thread both
 * call this, so they agree on which cluster's table holds which buckets. */
static inline void bs_block(
    uint32_t k, uint32_t blocks, uint32_t chunks, uint32_t* lo, uint32_t* hi
) {
    *lo = chunks * k / blocks;
    *hi = chunks * (k + 1) / blocks;
}
