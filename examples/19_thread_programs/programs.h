/* Thread programs example: what the X280 thread, the two programs' kernels and Python
 * share. Plain C. */
#pragma once

#include <stdint.h>

/* spin: SPIN, the core's share of units a0 to a0 + a1 - 1: unit u spins word 0 of costs
 * line u (a2) in cycles, then writes outs line u (a3): TP_MARK(u, cost, a4), u, a4. */
enum { TP_HANDLER_SPIN = 1 };

/* scale: SCALE multiplies the core's partition of a1 tiles from input tile a6 of the
 * input (a0) by a4 (float32 bits) into output tiles a3 + ... of the output (a2). */
enum { TP_HANDLER_SCALE = 1 };

/* The thread's handler: ROUNDS, a0 0 (the thread splits) or 1 (cores 0 to a1 - 1 for
 * spin in every round), a2 the rounds, a3 the cores, a4 the job, a5 scale's scalar. */
enum { TP_HANDLER_ROUNDS = 1 };

enum {
    TP_CB_IN = 0,
    TP_CB_OUT = 16,
    TP_CB_LINE = 1,
    TP_LINE_BYTES = 64,
    TP_TILE_BYTES = 2048
};

/* What a SPIN unit writes: mixed from the unit, its cost and the job. */
#define TP_MARK(unit, cost, job) \
    ((((unit) ^ (cost) << 7 ^ (job) << 26) * 0x9E3779B1u) ^ 0x85EBCA77u)

/* The thread's words, and each round's line in region `rounds` (from the host) and in
 * region `log` (from the thread). */
enum {
    TP_WORD_COSTS,
    TP_WORD_OUTS,
    TP_WORD_INPUT,
    TP_WORD_OUTPUT,
    TP_WORD_MAX_TILES, /* output tiles of a round */
    TP_WORDS
};

enum {
    TP_ROUND_SPIN,
    TP_ROUND_TILES,
    TP_ROUND_FIRST_UNIT,
    TP_ROUND_FIRST_TILE,
    TP_ROUND = 4
};

enum {
    TP_LOG_SPLIT,       /* k: spin on cores 0 to k - 1, scale on the rest */
    TP_LOG_SPIN_TICKS,  /* SPIN's event: submit to completion */
    TP_LOG_SCALE_TICKS, /* SCALE's */
    TP_LOG_ROUND_TICKS, /* the thread's round: decision, submits, both waits */
    TP_LOG_STATUS,      /* 0, or a KMX_ code that ended the job */
    TP_LOG = 8
};
