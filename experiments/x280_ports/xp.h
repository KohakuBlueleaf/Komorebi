/* X280 ports experiment: what the grid's kernels and the X280 thread share. Plain C. */
#pragma once

#include <stdint.h>

/* Segments. */
#define XP_SEG_TELL \
    1u /* an X280 core's tell: the core gives its word 0 as the result */
#define XP_SEG_SEND \
    2u                /* the core tells its X280 core a0 + its index, a1 the record's */
#define XP_SEG_ASK 3u /* a0 asks per core, each answered; a2 XP_F_*; stats page */
#define XP_SEG_BLOCK 4u /* a0 4 KiB blocks: a1 XP_BLOCK_L1 or _DRAM; a2 stats */
#define XP_SEG_RATE 5u  /* a0 1024-cycle units of tells both ways and timed asks */

/* Record words: a7 the round (the thread's), the asks' done tell carries it. */
#define XP_F_DONE 1u  /* the core tells its X280 core XP_DONE when its asks are done */
#define XP_F_READS 2u /* the writer reads DRAM pages while the reader asks */
#define XP_BLOCK_L1 1u
#define XP_BLOCK_DRAM 2u
#define XP_DONE 0x444F4E45u /* "DONE" */
#define XP_ASK_PLAIN 0u     /* ask word 0: what is asked */
#define XP_ASK_BLOCK 1u     /* w1: the L1 address of the core's page, w2: the round */
#define XP_ASK_DBLOCK 2u    /* w2: the round; answer: the block's bank address, tile */

/* The answer to a plain ask (k, core): checked by the core. */
#define XP_ANSWER(k, core) (((k) * 0x9E3779B1u) ^ ((core) << 8) ^ 0x5A5A0000u)
/* Word i of round r's block for core c. */
#define XP_BLOCK_WORD(r, c, i) ((((r) << 20) ^ ((c) << 12) ^ (i)) * 2654435761u)
#define XP_BLOCK_BYTES 4096u

/* The stats page of a core (32-bit words), XP_PAGE_BYTES per core. */
#define XP_PAGE_BYTES 1024u
#define XP_ST_COUNT 0u /* timed messages */
#define XP_ST_BAD 1u   /* answers or blocks that differ */
#define XP_ST_MIN 2u   /* cycles */
#define XP_ST_MAX 3u
#define XP_ST_SUM_LO 4u
#define XP_ST_SUM_HI 5u
#define XP_ST_SENT 6u  /* rate: tells the core sent */
#define XP_ST_SEEN 7u  /* rate: new X280 core tells the core saw */
#define XP_ST_READS 8u /* pages the writer read meanwhile */
#define XP_ST_ROUND 9u /* the record's a7 */
#define XP_ST_GAVE 10u /* nonzero: a wait gave up */
/* Bins: below 16 cycles one per cycle, then 8 per power of two (about 9 % wide) up to
 * 2^26 cycles, the last one open. */
#define XP_ST_HIST 32u
#define XP_HIST_BINS 192u

/* Scratch page (the program's buffer 16): block, stats, flags, writer's reads. */
#define XP_SCRATCH_BLOCK 0u
#define XP_SCRATCH_STATS 4096u
#define XP_SCRATCH_FLAGS                                         \
    5120u /* words: result seq, value; reader done; writer reads \
           */
#define XP_SCRATCH_READ 6144u
#define XP_READ_BYTES 2048u
#define XP_SCRATCH_BYTES 8192u

/* Thread modes (word 0) and the words after it. */
#define XP_MODE_TELL 1u /* rounds, core */
#define XP_MODE_SEND 2u /* rounds, core */
/* ASK: rounds, cores, asks, flags, stats, reads tensor, its pages, answers per X280
 * core */
#define XP_MODE_ASK 3u
#define XP_MODE_WINDOW 4u /* repeats, core */
#define XP_MODE_BLOCK 5u  /* rounds, cores, blocks, kind, bank 6 tile, bank 7, stats */
#define XP_MODE_RATE 6u   /* rounds, cores, units, stats */
#define XP_WORDS 10u
