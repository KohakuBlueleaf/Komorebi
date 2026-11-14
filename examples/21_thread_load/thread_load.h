/* Thread load example: what the X280 thread, the two programs' writers and Python
 * share. Plain C. */
#pragma once

#include <stdint.h>

/* sum and widen: example 16's readers and computes. Their writers here report one
 * line a core: the hash of its tiles, at line a3 + the core of the outs (a2). */
enum { TL_HANDLER_SUM = 1, TL_HANDLER_WIDEN = 1 };

/* The thread's handlers. ROUNDS: a0 the first round, a1 the rounds, a2 the job, a3
 * the boundary k when it starts (sum on cores 0 to k - 1, widen on k to the last). */
enum { TL_HANDLER_ROUNDS = 1, TL_HANDLER_CHECK = 2 };

/* CHECK: a0 widen's record, a1 a core of sum, a2 a core of widen, a3 the line of the
 * outs its commands write; its findings in region `check` (TL_CHECK_*). */
enum {
    TL_CHECK_REFUSED,     /* a SUM to widen's core: the event's kind << 8 | reason */
    TL_CHECK_BUSY,        /* a load to the core of a long SUM: the same */
    TL_CHECK_LOAD_STATUS, /* kmx_load of widen to that core: its return code */
    TL_CHECK_SUM_TICKS,   /* the long SUM: submit to completion */
    TL_CHECK_LOAD_TICKS,  /* from the long SUM's submit to kmx_load's return */
    TL_CHECK_BACK_STATUS, /* kmx_load of sum back to it */
    TL_CHECK_AFTER,       /* a SUM to it after both loads: the event's kind */
    TL_CHECK_LONG_TILES,  /* the long SUM's tiles */
    TL_CHECK = 16
};

/* Buffers the writers use for the line they write (each program its own). */
enum { TL_CB_SUM_LINE = 2, TL_CB_WIDEN_LINE = 24, TL_LINE_BYTES = 64 };

/* A core's line: the hash (sum: its tile's; widen: the sum of its tiles' hashes), its
 * tiles, the command's a4 (job << 16 | round), the program, the first input tile. */
enum {
    TL_LINE_HASH,
    TL_LINE_TILES,
    TL_LINE_TAG,
    TL_LINE_PROGRAM,
    TL_LINE_FIRST,
    TL_LINE_WORDS = 16
};

enum { TL_PROGRAM_SUM = 1, TL_PROGRAM_WIDEN = 2 };

/* A tile's hash: from TL_HASH_START, each 32-bit word of the tile in L1 order. */
enum { TL_HASH_START = 0x2545F491 };

#define TL_MIX(hash, word) ((((hash) << 5) | ((hash) >> 27)) + (word))

/* The thread's words, each round's line in region `rounds` (from the host) and in
 * region `log` (from the thread). */
enum { TL_WORD_INPUT, TL_WORD_OUTS, TL_WORD_CORES, TL_WORDS };

enum {
    TL_ROUND_SUM,       /* tiles SUM adds over sum's cores */
    TL_ROUND_WIDEN,     /* tiles WIDEN writes over widen's cores */
    TL_ROUND_SUM_FIRST, /* their first input tiles */
    TL_ROUND_WIDEN_FIRST,
    TL_ROUND = 4
};

enum {
    TL_LOG_SPLIT,       /* k: sum on cores 0 to k - 1, widen on the rest */
    TL_LOG_LOADS,       /* loads issued in the round (0 or 1) */
    TL_LOG_LOAD_TICKS,  /* kmx_load's call: issue to done, the thread's clock */
    TL_LOG_LOAD_EVENT,  /* the load's event: the dispatcher's submit to completion */
    TL_LOG_SUM_TICKS,   /* SUM's event */
    TL_LOG_WIDEN_TICKS, /* WIDEN's event */
    TL_LOG_ROUND_TICKS, /* the round: decision, load, both commands and waits */
    TL_LOG_STATUS,      /* 0, or the KMX_ code that ended the job */
    TL_LOG = 8
};

/* The boundary for a round: sum's share of the tiles of the cores, 1 to cores - 1. */
#define TL_SPLIT(cores, sum, widen) \
    ((cores) * (sum) + ((sum) + (widen)) / 2) / ((sum) + (widen))
