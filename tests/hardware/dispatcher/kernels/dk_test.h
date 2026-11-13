/* The stand-in worker's test handlers, argument words and per-core counters, shared by
 * the kernel (C++) and mirrored by the driver (dispatch_card.py). */
#pragma once

#include <stdint.h>

/* Handlers besides KM_HANDLER_NOP and KM_HANDLER_EXIT. */
#define DK_HANDLER_CHECK 1u /* check a2..a9 against a0 (key) and a1 (parameter) */
#define DK_HANDLER_DELAY 2u /* wait a1 Tensix clock cycles */
#define DK_HANDLER_STALL 3u /* core a1 waits for the stop flag; the rest run a NOP */
/* result dk_result(a0, a1); with KM_FLAG_RESULT the leader sends it */
#define DK_HANDLER_RESULT 4u
#define DK_HANDLER_SUM 5u /* result a0 + count (the graph references' result word) */

/* Argument word w (2..9) of a command with key a0 and parameter a1. */
static inline uint32_t dk_arg(uint32_t key, uint32_t param, uint32_t w) {
    uint32_t x = key * 0x9E3779B1u ^ param * 0x85EBCA77u ^ w * 0xC2B2AE3Du;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    return x;
}

/* The result of a DK_HANDLER_RESULT command with key a0 and parameter a1. */
static inline uint32_t dk_result(uint32_t key, uint32_t param) {
    return dk_arg(key, param, 10u);
}

/* A key's contribution to a core's key mix (XOR over the commands it ran). */
static inline uint32_t dk_mix(uint32_t key) {
    return dk_arg(key, 0x5EEDu, 0u);
}

/* Gave-up codes and where codes of the state words. */
#define DK_GAVE_STOP 1u  /* the stop flag was set */
#define DK_GAVE_LIMIT 2u /* the wait's own timeout ran out */
#define DK_GAVE_NOC 3u   /* a NoC transfer did not complete within the timeout */
#define DK_GAVE_ALIGN 4u /* an address from the host is not 16-byte aligned */
#define DK_WHERE_BOOT 1u /* a leader collecting its column's check-ins */
#define DK_WHERE_NEXT 2u /* waiting for a command */
#define DK_WHERE_DONE 3u /* a group leader waiting for its cores */
#define DK_WHERE_STALL 4u
#define DK_WHERE_DELAY 5u
#define DK_WHERE_EXIT 6u       /* left on an EXIT command */
#define DK_WHERE_NOC_READ 7u   /* waiting for reads to arrive */
#define DK_WHERE_NOC_WRITE 8u  /* waiting for writes to be acknowledged */
#define DK_WHERE_NOC_ATOMIC 9u /* waiting for atomics to be acknowledged */

/* State words at KM_CONTROL_STATE: last sequence consumed per slot, then these. */
#define DK_S_GAVE 16u
#define DK_S_WHERE 17u
#define DK_S_SEQUENCE 18u /* the command being run, or the last one */

/* Per-core counters: DK_WORDS words, page `me` of the output tensor. */
enum {
    DK_C_MAGIC = 0,      /* DK_MAGIC | core index: the kernel reached its end */
    DK_C_TAG = 1,        /* the tag it ran with */
    DK_C_RUN = 2,        /* commands run in a group (EXIT not counted) */
    DK_C_KEY_SUM = 3,    /* sum of their keys a0 */
    DK_C_KEY_MIX = 4,    /* XOR of dk_mix(a0) */
    DK_C_IGNORED = 5,    /* commands seen that were not this core's */
    DK_C_ARG_ERRORS = 6, /* argument words that did not match */
    DK_C_HDR_ERRORS = 7, /* slot field not the slot, or an unknown handler */
    DK_C_EXIT = 8,       /* sequence of the EXIT command, 0 if none */
    DK_C_ACKS = 9,       /* acknowledgements pushed as a group leader */
    DK_C_INCS = 10,      /* completions reported to a leader */
    DK_C_HANDLER = 11, /* commands run per handler: NOP, CHECK, DELAY, STALL, RESULT */
    DK_C_STATE = 16,   /* copy of the 19 state words */
    DK_WORDS = 64
};

#define DK_MAGIC 0xD15C0000u
