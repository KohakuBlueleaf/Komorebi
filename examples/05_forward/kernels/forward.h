// The forward example: its handler, message words, command arguments and per-core
// results.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerForward = 1;  // the dynamic handler

// Message words (64 bytes); word 0 is the runtime's mark.
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kMsgId = 1;
constexpr uint32_t kMsgForwards = 2;  // forwards left
constexpr uint32_t kMsgState = 3;     // the generator
constexpr uint32_t kMsgSum = 4;       // sum of the states it visited with (mod 2**32)
constexpr uint32_t kNullId = 0xFFFFFFFF;  // quota probe messages: consume at once

// RUN command: a0 message_count, a1 forwards per message, a2 mode, a3 target_rank (mode
// 1), a4 stall_rank_plus_one (0: none), a5 probe_rank_plus_one (0: none).
constexpr uint32_t kModeRandom = 0;   // next core: state mod group size
constexpr uint32_t kModeOneCore = 1;  // every forward goes to rank a3 target_rank

// Per-core results, 16 words at the address of compute argument 2.
constexpr uint32_t kResConsumed = 0;
constexpr uint32_t kResTotalLo = 1;  // sum of the consumed messages' sums, 64 bits
constexpr uint32_t kResTotalHi = 2;
constexpr uint32_t kResVisits = 3;
constexpr uint32_t kResLastConsume = 4;  // wall clock (low word) of the last consume
constexpr uint32_t kResNull = 5;         // quota probe messages consumed
constexpr uint32_t kResSent = 6;         // probe sends that returned true
constexpr uint32_t kResRejected = 7;     // sends that returned false
constexpr uint32_t kResQuota = 8;        // quota left at the end
constexpr uint32_t kResRun = 9;  // the run's sequence: the results are this run's
constexpr uint32_t kResWords = 16;

// The generator (xorshift32) and a message's first state.
inline uint32_t forward_next(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline uint32_t forward_seed(uint32_t id) {
    const uint32_t state = (id + 1) * 0x9E3779B1u;
    return state == 0 ? 1 : state;
}
