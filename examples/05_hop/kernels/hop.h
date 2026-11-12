// The hop example: its segment, item words, record arguments and per-core results.
#pragma once

#include <stdint.h>

constexpr uint32_t kSegHop = 1;  // the dynamic segment

// Item words (64 bytes); word 0 is the runtime's mark.
constexpr uint32_t kItemWords = 16;
constexpr uint32_t kItId = 1;
constexpr uint32_t kItHops = 2;   // hops left
constexpr uint32_t kItState = 3;  // the generator
constexpr uint32_t kItSum = 4;    // sum of the states it visited with (mod 2**32)
constexpr uint32_t kNullId = 0xFFFFFFFF;  // quota probe items: retire at once

// RUN record: a0 items, a1 hops per item, a2 mode, a3 target member (mode 1),
// a4 1 + member that stalls in the body (0: none), a5 1 + member that probes its quota.
constexpr uint32_t kModeRandom = 0;   // next core: state mod team size
constexpr uint32_t kModeOneCore = 1;  // every hop goes to member a3

// Per-core results, 16 words at the address of compute argument 2.
constexpr uint32_t kResRetired = 0;
constexpr uint32_t kResTotalLo = 1;  // sum of the retired items' sums, 64 bits
constexpr uint32_t kResTotalHi = 2;
constexpr uint32_t kResVisits = 3;
constexpr uint32_t kResLastRetire = 4;  // wall clock (low word) of the last retire
constexpr uint32_t kResNull = 5;        // quota probe items retired
constexpr uint32_t kResSpawned = 6;     // probe spawns that returned true
constexpr uint32_t kResRefused = 7;     // spawns that returned false
constexpr uint32_t kResQuota = 8;       // quota left at the end
constexpr uint32_t kResRun = 9;  // the run's sequence: the results are this run's
constexpr uint32_t kResWords = 16;

// The generator (xorshift32) and an item's first state.
inline uint32_t hop_next(uint32_t s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

inline uint32_t hop_seed(uint32_t id) {
    const uint32_t s = (id + 1) * 0x9E3779B1u;
    return s == 0 ? 1 : s;
}
