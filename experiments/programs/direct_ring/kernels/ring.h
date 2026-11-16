// The direct ring: its handler, token words, command arguments, per-core results and
// the next rank of a token (the host's replay has the same functions).
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerRing = 1;  // the dynamic handler

// Token words (64 bytes); word 0 is the runtime's mark.
constexpr uint32_t kTokenWords = 16;
constexpr uint32_t kTokId = 1;
constexpr uint32_t kTokHops = 2;   // hops left
constexpr uint32_t kTokState = 3;  // the generator
constexpr uint32_t kTokSum = 4;    // sum of the ranks it visited (mod 2**32)

// RUN command: a0 tokens per rank, a1 hops per token, a2 mode, a3 pattern, a4 ranks
// of the pattern (from rank 0; the others seed nothing and get nothing).
constexpr uint32_t kModeQueue = 0;     // every hop a forward through the queues
constexpr uint32_t kModeDirect = 1;    // a direct message, a forward when it is busy
constexpr uint32_t kPatternRing = 0;   // rank r to r + 1 (an offset)
constexpr uint32_t kPatternPairs = 1;  // rank r to one of ring_pair(r, 0 or 1)

// Per-core results: counts, then a record of kRecordWords for each token ended here.
constexpr uint32_t kResVisits = 0;
constexpr uint32_t kResConsumed = 1;
constexpr uint32_t kResDirect = 2;  // hops that went direct
constexpr uint32_t kResBusy = 3;    // direct() false: forwarded instead
constexpr uint32_t kResRun = 4;     // the run's sequence: the results are this run's
constexpr uint32_t kResRecords = 8;
constexpr uint32_t kRecordWords = 4;  // id + 1, state, sum, hops left (0)

inline uint32_t ring_next(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint32_t ring_seed(uint32_t id) {
    const uint32_t state = (id + 1) * 0x9E3779B1u;
    return state == 0 ? 1 : state;
}

// The two declared receivers of rank r in the pairs pattern (equal: one pair).
inline uint32_t ring_pair(uint32_t rank, uint32_t which, uint32_t ranks) {
    return which == 0 ? (5 * rank + 1) % ranks : (rank * rank + 3) % ranks;
}

// The rank a token with state `state` goes to from `rank`.
inline uint32_t ring_to(
    uint32_t rank, uint32_t state, uint32_t pattern, uint32_t ranks
) {
    if (pattern == kPatternRing) {
        return (rank + 1) % ranks;
    }
    return ring_pair(rank, state >> 7 & 1, ranks);
}
