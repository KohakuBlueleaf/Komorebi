/* X280 ports example: what the grid's reader and the X280 thread share. Plain C. */
#pragma once

#include <stdint.h>

/* a0 asks per core; the core first waits for its X280 core's salt */
#define HX_HANDLER_WORK 1u

/* The salt an X280 core tells core c in round r, and its answer to the core's ask k. */
#define HX_SALT(r, c) (((r) * 0x9E3779B1u) ^ ((c) * 0x85EBCA77u))
#define HX_VALUE(k, c) (((k) * 0xC2B2AE3Du) ^ ((c) << 16) ^ 0x27D4EB2Fu)
/* The core's running hash: from the salt, one step per answer. */
#define HX_MIX(acc, v) (((acc) ^ (v)) * 0x01000193u)

/* Thread words (cores 0: every core of the grid), and the line each X280 core keeps per
 * core it serves in region `out`. */
enum { HX_W_ROUNDS, HX_W_CORES, HX_W_ASKS, HX_WORDS };

enum {
    HX_L_HASH,   /* the core's hash of the last round */
    HX_L_ROUNDS, /* rounds whose hash came */
    HX_L_CALLS,  /* asks the core timed */
    HX_L_SUM_LO, /* their cycles (AICLK), 64 bits */
    HX_L_SUM_HI,
    HX_L_MAX,
    HX_L_WHO, /* 1 + (cluster << 2 | cpu) */
    HX_LINE = 16
};

/* The pull mode (pull.c): a pass over units 0 to a0 - 1, unit u busy for HX_COST cycles
 * (a1: 0 equal ranges, 1 pulled ranges); a report tells the pass's numbers back. */
#define HX_HANDLER_PASS 2u   /* a0 to a5: units, how, base, extra, centre, half */
#define HX_HANDLER_REPORT 3u /* a0 0: units, work, span; 1: hash, pulls, waited */

/* Unit u's cycles: `base`, plus up to `extra` in a triangle of half-width `half` about
 * `centre` (extra * half below 2^32). */
#define HX_DIST(u, c) ((u) > (c) ? (u) - (c) : (c) - (u))
#define HX_COST(u, base, extra, centre, half)                                        \
    ((base) +                                                                        \
     (HX_DIST(u, centre) < (half) ? (extra) * ((half) - HX_DIST(u, centre)) / (half) \
                                  : 0u))
/* A unit's mark in the core's hash: a sum over its units, the same in any order. */
#define HX_MARK(u) (((u) * 0x9E3779B1u ^ ((u) * 0x9E3779B1u) >> 15) * 0x85EBCA77u)

/* The pull thread's words, and the lines it keeps per kind of pass and core in `out`.
 */
enum { HP_W_UNITS, HP_W_PER, HP_W_BASE, HP_W_EXTRA, HP_W_CENTRE, HP_W_HALF, HP_WORDS };

enum { HP_EQUAL, HP_PULL, HP_LOG, HP_REPLAY, HP_PICK, HP_KINDS };

enum {
    HP_L_UNITS,  /* units the core ran */
    HP_L_WORK,   /* cycles (AICLK) it was busy on them */
    HP_L_SPAN,   /* cycles from the pass's start to its end on the core */
    HP_L_HASH,   /* the sum of its units' HX_MARK */
    HP_L_PULLS,  /* ranges it pulled */
    HP_L_WAITED, /* cycles it waited in next() */
    HP_L_WHO,    /* 1 + (cluster << 2 | cpu) */
    HP_LINE = 8
};
