// Leader example: handlers, buffers and arguments its kernels share, the leader's
// choice from a value and the four steps a member can take.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"

// LEAD: a0 values_address, a1 casts, a2 results_address, a3 first (index of the
// first value), a4 result_page (page of rank 0). Words alone, broadcast 0.
constexpr uint32_t kHandlerLead = 1;
// CARRY: as LEAD, and each cast also carries tile `index % a6` of a5 tiles_address
// (broadcast 1); a7 sums_address takes each core's sum of its tiles, page a4 + rank.
constexpr uint32_t kHandlerCarry = 2;

constexpr uint32_t kCbTiles = 0;   // broadcast 1's buffer: window blocks of one tile
constexpr uint32_t kCbValues = 1;  // the leader's page of values
constexpr uint32_t kCbResult = 2;  // a core's result row, written to DRAM
constexpr uint32_t kCbSum = 16;    // one fp32 tile: the sum of the tiles received
constexpr uint32_t kValuesPage = 256;  // values (uint32) per page
constexpr uint32_t kResultWords = 16;  // words of a result row: state, wrong, first wrong
constexpr uint32_t kNone = 0xFFFFFFFFu;

// Reader arguments: broadcast 0's words, then broadcast 1's. Compute: B.
constexpr uint32_t kArgWords = 0;
constexpr uint32_t kArgTiles = km::kBroadcastArgs;

// The leader's choice from a value it read: which quarter of the range it falls in.
inline uint32_t pick(uint32_t value) {
    return value >> 30;
}

// The word cast: the choice in the low 2 bits, the value's index + 1 above it.
inline uint32_t choice_word(uint32_t choice, uint32_t index) {
    return choice | (index + 1) << 2;
}

// The four things a core can do next, compiled in; the leader's word picks one.
inline uint32_t step(uint32_t choice, uint32_t state, uint32_t index) {
    switch (choice & 3u) {
        case 0:
            return state * 33u + index;
        case 1:
            return state ^ state << 7;
        case 2:
            return state + 0x9E3779B9u;
        default:
            return state >> 3 | state << 29;
    }
}
