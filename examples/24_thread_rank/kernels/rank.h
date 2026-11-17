/* Example 24: what the cores' compute kernel and the X280 thread share. Plain C. */
#pragma once

#include <stdint.h>

/* Handlers, both dynamic: GRID is the cores' body (the RUN names it), X280 the X280
 * ranks'. A message carries the one that handles it where it lands. */
#define kHandlerGrid 1u
#define kHandlerX280 2u

/* Message words (64 bytes); word 0 is the runtime's. */
#define kMessageWords 16u
#define kMsgId 1u
#define kMsgHops 2u  /* visits left before the one that consumes it */
#define kMsgValue 3u /* stepped at every visit */

/* RUN: a0 messages each core seeds, a1 the hops of an even message (odd: one more),
 * a2 the seeding's flags (0: each seed to an X280 rank, odd ids one hop more): */
#define kSeedToCores 1u /* each seed to the core value % count first */
#define kSeedEven 2u    /* odd ids no hop more */

/* Per-core results, 16 words at compute argument 0's address. */
#define kResConsumed 0u /* messages consumed here */
#define kResSumLo 1u    /* sum of their values, 64 bits */
#define kResSumHi 2u
#define kResXor 3u    /* xor of id * 0x9E3779B1 + value */
#define kResVisits 4u /* messages handled here, the consumed ones included */
#define kResToX280 5u /* sends and forwards to the X280 ranks */
#define kResRun 6u    /* the run's sequence */
#define kResRejected 7u
#define kResWords 16u

/* The X280 ranks' totals in the thread's region 0: rank r's at word r * kX280Words. */
#define kX280Consumed 0u
#define kX280SumLo 1u
#define kX280SumHi 2u
#define kX280Xor 3u
#define kX280Takes 4u /* messages taken from its queue */
#define kX280Sent 5u  /* forwards to cores */
#define kX280Gave 6u  /* KMX_GAVE_*, 0: none */
#define kX280Quota 7u /* sends at the run's start: max_live // the run's ranks */
#define kX280Words 8u

/* The value's step at a core and at an X280 rank, and a message's first value. */
static inline uint32_t grid_step(uint32_t value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

static inline uint32_t x280_step(uint32_t value, uint32_t rank) {
    return value * 0x2C1B3C6Du + 0x297A2D39u + rank;
}

static inline uint32_t seed_value(uint32_t id) {
    const uint32_t value = (id + 1u) * 0x9E3779B1u;
    return value == 0 ? 1u : value;
}

static inline uint32_t end_mix(uint32_t id, uint32_t value) {
    return id * 0x9E3779B1u + value;
}
