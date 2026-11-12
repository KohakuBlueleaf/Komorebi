/* Work-cluster example: the decision in plain C, built into the X280 thread and, for
 * the host-fed path, into a host library; both must issue the same records. */
#pragma once

#include <stdint.h>

/* STEP: a0 state, a1 round + 1: x[p] += a1 + p on the cores of the team. */
#define WC_SEG_STEP 1u
/* SUM: a0 state, a1 cores: the leader's result word is x[0] + ... + x[a1 - 1]. */
#define WC_SEG_SUM 2u

/* The program's words, in this order. */
enum { WC_ADDR, WC_CORES, WC_COUNT, WC_LIMIT, WC_MAX_ROUNDS, WC_WORDS };

/* What wc_result returns. */
enum { WC_AGAIN = 0, WC_END = 1, WC_NO_END = 2 };

typedef struct {
    uint32_t word[WC_WORDS];
    uint32_t round; /* rounds finished */
    uint32_t first; /* the next STEP's first core */
    uint32_t sum;   /* the last SUM's result word */
    uint32_t trail; /* a hash of every finished round's STEP team */
} wc_state;

/* One record: segment, flags (KM_F_EXCL, KM_F_RESULT), team, two arguments. */
typedef struct {
    uint32_t seg, flags, first, count, a0, a1;
} wc_record;

/* 0, or 1 when the words cannot make a run (a team outside the grid, no rounds). */
uint32_t wc_start(wc_state* s, const uint32_t* words);

/* The round's records: STEP on `count` cores from the core the last sum chose, then
 * SUM over every core, exclusive (it waits for STEP), with a result word. */
void wc_round(const wc_state* s, wc_record out[2]);

/* SUM's result word: WC_END when it reached the limit, WC_NO_END after the last
 * allowed round, else WC_AGAIN with the next team chosen. */
uint32_t wc_result(wc_state* s, uint32_t sum);
