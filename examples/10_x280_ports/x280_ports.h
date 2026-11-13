/* X280 ports example: what the grid's reader and the X280 thread share. Plain C. */
#pragma once

#include <stdint.h>

/* WORK: a0 asks (per core); the core first waits for its X280 core's salt */
#define HX_HANDLER_WORK 1u

/* The salt an X280 core tells a Tensix core in a round, and its answer to the core's
 * ask (counted from 1). */
#define HX_SALT(round, core) (((round) * 0x9E3779B1u) ^ ((core) * 0x85EBCA77u))
#define HX_VALUE(ask, core) (((ask) * 0xC2B2AE3Du) ^ ((core) << 16) ^ 0x27D4EB2Fu)
/* The core's running hash: from the salt, one step per answer. */
#define HX_MIX(hash, value) (((hash) ^ (value)) * 0x01000193u)

/* Thread words (cores 0: every core of the grid), and the line each X280 core keeps per
 * core it serves in region `lines`. */
enum { HX_WORD_ROUNDS, HX_WORD_CORES, HX_WORD_ASKS, HX_WORDS };

enum {
    HX_LINE_HASH,    /* the core's hash of the last round */
    HX_LINE_ROUNDS,  /* rounds whose hash came */
    HX_LINE_CALLS,   /* asks the core timed */
    HX_LINE_SUM_LOW, /* their cycles (AICLK), 64 bits */
    HX_LINE_SUM_HIGH,
    HX_LINE_LARGEST,
    HX_LINE_SERVED_BY, /* 1 + (cluster << 2 | x280_core) */
    HX_LINE = 16
};

/* The pull mode (pull.c): a pass over units, each busy for HX_COST cycles; a report
 * tells the pass's numbers back. */

/* PASS: a0 units, a1 pulled (0 equal ranges, 1 pulled ranges), a2 base, a3 extra, a4
 * centre, a5 half. */
#define HX_HANDLER_PASS 2u
/* REPORT: a0 page (0: units, work, span; 1: hash, pulls, waited). */
#define HX_HANDLER_REPORT 3u

/* A unit's cycles: `base`, plus up to `extra` in a triangle of half-width `half` about
 * `centre` (extra * half below 2^32). */
#define HX_DIST(unit, centre) \
    ((unit) > (centre) ? (unit) - (centre) : (centre) - (unit))
#define HX_COST(unit, base, extra, centre, half)                \
    ((base) +                                                   \
     (HX_DIST(unit, centre) < (half)                            \
          ? (extra) * ((half) - HX_DIST(unit, centre)) / (half) \
          : 0u))
/* A unit's mark in the core's hash: a sum over its units, the same in any order. */
#define HX_MARK(unit) \
    (((unit) * 0x9E3779B1u ^ ((unit) * 0x9E3779B1u) >> 15) * 0x85EBCA77u)

/* The pull thread's words, and the lines it keeps per kind of pass and core in `lines`.
 */
enum {
    HP_WORD_UNITS,
    HP_WORD_UNITS_PER_RANGE,
    HP_WORD_BASE,
    HP_WORD_EXTRA,
    HP_WORD_CENTRE,
    HP_WORD_HALF,
    HP_WORDS
};

enum { HP_EQUAL, HP_PULL, HP_LOG, HP_REPLAY, HP_PICK, HP_KINDS };

enum {
    HP_LINE_UNITS,     /* units the core ran */
    HP_LINE_WORK,      /* cycles (AICLK) it was busy on them */
    HP_LINE_SPAN,      /* cycles from the pass's start to its end on the core */
    HP_LINE_HASH,      /* the sum of its units' HX_MARK */
    HP_LINE_PULLS,     /* ranges it pulled */
    HP_LINE_WAITED,    /* cycles it waited in next() */
    HP_LINE_SERVED_BY, /* 1 + (cluster << 2 | x280_core) */
    HP_LINE = 8
};
