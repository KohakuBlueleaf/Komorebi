/* Remote loading from an X280 core: what the grid's reader, the thread and the host
 * share (the message, the region layouts, the thread words). Plain C. */
#pragma once

#include <stdint.h>

#define RL_HANDLER_SWAP 1u /* the reader serves messages until RL_END */

/* A message: w0 DRAM address, w1 RL_LOAD (the core reads it) | bank << 24 | bytes,
 * w2 the argument; w1 == RL_END ends the handler. */
#define RL_LOAD 0x00800000u
#define RL_BYTES 0x001FFFFFu
#define RL_END 0xFFFFFFFFu
/* The core's reply: w0 the result, w1 its call counter, w2 load << 16 | run cycles
 * (each at most 0xFFFF); on an error w1 is RL_FAIL and w2 the reason. */
#define RL_FAIL 0xFFFFFFFFu
#define RL_ERR_SIZE 1u
#define RL_ERR_READ 2u

/* Thread modes: who moves the bytes. */
enum {
    RL_MODE_SELF = 1,  /* the core reads the body from DRAM (the X280 decides) */
    RL_MODE_STORE = 2, /* the X280 core stores the body into the slot */
    RL_MODE_PATCH = 3, /* the X280 core stores the template once, then its constant */
};

/* Thread words. */
enum {
    RL_W_MODE,
    RL_W_SWAPS,    /* per core */
    RL_W_CORES,    /* cores 0..n-1 */
    RL_W_SLOT,     /* the slot's L1 address on every core */
    RL_W_IMG_BASE, /* the DRAM images: page p in bank p % 8 at base + (p / 8) * page */
    RL_W_IMG_PAGE,
    RL_W_FIRST,   /* the run's bodies: first + schedule entry */
    RL_W_MARK_AT, /* RL_MODE_PATCH: byte offset of the lui / addi pair in the body */
    RL_W_MARK_RD, /* their destination register */
    RL_WORDS
};

/* Region `cfg` (words): per body its bytes and its byte offset in region `bodies`,
 * each X280 core's schedule, each core's NoC x | y << 16. */
#define RL_BODIES_MAX 8u
#define RL_SCHED 8u
#define RL_CFG_BYTES 0u
#define RL_CFG_OFFSET (RL_CFG_BYTES + RL_BODIES_MAX)
/* the schedules: [cluster * 4 + cpu][RL_SCHED] */
#define RL_CFG_SCHED (RL_CFG_OFFSET + RL_BODIES_MAX)
#define RL_CFG_NOC (RL_CFG_SCHED + 16u * RL_SCHED)
#define RL_CFG_WORDS (RL_CFG_NOC + 128u)

/* Region `out`: a line per core and swap (core * swaps + k). */
enum {
    RL_L_RESULT,
    RL_L_CALLS,
    RL_L_CYCLES, /* load << 16 | run */
    RL_L_TICKS,  /* the X280 core's ticks from deciding to the reply */
    RL_L_STORE,  /* ticks of its store into the slot */
    RL_L_BODY,
    RL_L_ARG,
    RL_L_WHO, /* 1 + (cluster << 2 | cpu) */
    RL_LINE = 8
};

#define RL_LINES_MAX 12800u

/* Core c's argument of swap k, and the value the template's constant takes. */
#define RL_ARG(c, k) (((uint32_t)(c) << 20) ^ ((uint32_t)(k) * 0x9E3779B1u))
#define RL_VALUE(k) ((uint32_t)(k) * 0x9E3779B9u + 0x1234u)
