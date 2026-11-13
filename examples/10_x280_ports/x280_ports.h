/* X280 ports example: what the grid's reader and the X280 thread share. Plain C. */
#pragma once

#include <stdint.h>

#define HX_SEG_WORK                                                       \
    1u /* a0 asks per core; the core first waits for its X280 core's salt \
        */

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
