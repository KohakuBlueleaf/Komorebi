/* Work-cluster example: the decision in plain C, built into the X280 thread and, for
 * the host-fed path, into a host library; both must submit the same commands. */
#pragma once

#include <stdint.h>

/* STEP: a0 state_address, a1 round (this round + 1): x[p] += round + p on the cores
 * of the group. */
#define WC_HANDLER_STEP 1u
/* SUM: a0 state_address, a1 cores: the leader (core `first`) returns x[first] + ... +
 * x[first + cores - 1]. */
#define WC_HANDLER_SUM 2u

/* The program's words, in this order: the loop runs on cores [base, base + cores). */
enum { WC_ADDR, WC_CORES, WC_COUNT, WC_LIMIT, WC_MAX_ROUNDS, WC_BASE, WC_WORDS };

/* What wc_result returns. */
enum { WC_AGAIN = 0, WC_END = 1, WC_NO_END = 2 };

typedef struct {
    uint32_t word[WC_WORDS];
    uint32_t round; /* rounds finished */
    uint32_t first; /* the next STEP's first core */
    uint32_t sum;   /* the last SUM's result word */
    uint32_t trail; /* a hash of every finished round's STEP group */
} wc_state;

/* One command: handler, flags (KM_FLAG_EXCLUSIVE, KM_FLAG_RESULT), group, two
 * arguments. */
typedef struct {
    uint32_t handler, flags, first, count, a0, a1;
} wc_command;

/* 0, or 1 when the words cannot make a run (a group outside the grid, no rounds). */
uint32_t wc_start(wc_state* state, const uint32_t* words);

/* The round's commands: STEP on `count` cores from the core the last sum chose, then
 * SUM over the loop's cores, exclusive (it waits for STEP), with a result word. */
void wc_round(const wc_state* state, wc_command commands[2]);

/* SUM's result word: WC_END when it reached the limit, WC_NO_END after the last
 * allowed round, else WC_AGAIN with the next group chosen. */
uint32_t wc_result(wc_state* state, uint32_t sum);
