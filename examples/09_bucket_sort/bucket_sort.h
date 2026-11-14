/* Bucket sort: what the kernels (C++) and the thread (C) share. The host script
 * follows the same words; README.md explains every one. */
#pragma once

#include <stdint.h>

/* Step 1, COUNT: every core counts its partition of the keys per bucket and writes its
 * row of the counts table. Step 3, PLACE: every core writes each key to its place. */
#define BS_HANDLER_COUNT 1u
#define BS_HANDLER_PLACE 2u

/* The thread's own command, built with BS_STARTED_BY_HOST (islands): the rounds into
 * the output a0 names, the result words to row a1 of region `results`. */
#define BS_THREAD_SORT 1u

/* The words of both commands: a0 keys_address, a1 key_count, a2 log2_buckets |
 * blocks << 8, a3 banks (8 bits per block), a4 to a6 tables, a7 output_address. */

/* A bucket is the key's top log2(buckets) bits; buckets go in chunks of 16, one line
 * of a table row, so that no line or NoC transfer holds two blocks' buckets. */
#define BS_CHUNK 16u
#define BS_BLOCKS_MAX 3u       /* one block of buckets per work cluster taking part */
#define BS_BUCKETS_MAX 16384u  /* a core keeps a counter per bucket in L1 */
#define BS_KEYS_PER_PAGE 1024u /* the keys and output tensors: 4 KiB pages */
#define BS_SLOTS 64u           /* PLACE: 16-byte staging slots for the 4-byte writes */

/* The thread's words, in this order (the host script writes them). */
enum {
    BS_WORD_KEY_COUNT,      /* the keys to sort */
    BS_WORD_LOG2_BUCKETS,   /* log2(buckets), 4 to 14 */
    BS_WORD_CORES,          /* the Tensix group: cores 0 to cores - 1 */
    BS_WORD_KEYS_ADDRESS,   /* the keys tensor's DRAM address */
    BS_WORD_OUTPUT_ADDRESS, /* the output tensor's DRAM address */
    BS_WORD_X280_CORES,     /* 1, 4 or 12 */
    BS_WORD_REPEATS,        /* rounds of count, plan, place (the best time is kept) */
    BS_WORD_BANKS,          /* the DRAM bank of work cluster c in bits 8 (c - 1) */
    BS_WORDS
};

/* The thread's result words: the best time of each step in timer ticks (50 per
 * microsecond), the keys the plan placed (must be the key count), the rounds done. */
enum {
    BS_RESULT_COUNT,
    BS_RESULT_PLAN,
    BS_RESULT_PLACE,
    BS_RESULT_WHOLE,
    BS_RESULT_PLACED,
    BS_RESULT_ROUNDS,
    BS_RESULT_WORDS
};

/* Block `block` of `blocks`: chunks [first_chunk, end_chunk) of `chunks`; kernels and
 * thread both call this, so they agree on which table holds which buckets. */
static inline void bs_block(
    uint32_t block,
    uint32_t blocks,
    uint32_t chunks,
    uint32_t* first_chunk,
    uint32_t* end_chunk
) {
    *first_chunk = chunks * block / blocks;
    *end_chunk = chunks * (block + 1) / blocks;
}
