/* Commands to an X280 thread: the handlers of the grid and of the thread, and the word
 * the thread's echo returns (the host computes the same). Plain C. */
#pragma once

#include <stdint.h>

/* The grid's: the group's leader gives tc_grid(a0) as its result word. */
#define KM_TC_GRID_RESULT 1u
/* The thread's (kmx_program): */
#define KM_TC_ECHO 10u /* returns tc_echo(a0, a1) */
#define KM_TC_SPIN 11u /* waits a0 timer ticks (or a stop), returns a1 */
/* KM_TC_GRID_RESULT of a0 on core a1, waited for; returns its word + 1 */
#define KM_TC_SUBMIT 12u
/* KM_TC_SUBMIT's word when the command did not run */
#define KM_TC_FAILED 0xFFFFFFFFu

/* The client thread's words. */
#define KM_TC_WORD_COUNT 0u   /* commands of each kind */
#define KM_TC_WORD_CLUSTER 1u /* the cluster whose thread it commands */
#define KM_TC_WORD_CORE 2u    /* the grid core of its NOP commands */
#define KM_TC_WORDS 3u

static inline uint32_t tc_echo(uint32_t a0, uint32_t a1) {
    return ((a0 * 0x9E3779B1u) ^ a1) + 0x7F4A7C15u;
}

static inline uint32_t tc_grid(uint32_t a0) {
    return a0 ^ 0xA5A5A5A5u;
}
