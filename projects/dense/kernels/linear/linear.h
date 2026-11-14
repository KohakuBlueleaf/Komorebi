// Linear layer Y = X x W written for islands: each island computes with the part of W
// it holds, the grid plan inside it. Handlers, buffers, arguments and ports follow.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_placement.hpp"

// LINEAR: a0 x_address (X, [M, K] tiles), a1 w_address (W, [K, N]), a2 y_address (Y,
// [M, N]), a3 p_address (P, [M, N] fp32: an island's partial sums), a4 k_tiles (K).
constexpr uint32_t kHandlerLinear = 1;

// Buffers: the grid plan's (row and column broadcasts, bf16 output subblocks, fp32
// partial sums), then fp32 output subblocks of a partial product, and the shares.
constexpr uint32_t kCbRow = 0;
constexpr uint32_t kCbCol = 1;
constexpr uint32_t kCbYOut = 16;
constexpr uint32_t kCbPart = 24;
constexpr uint32_t kCbPartOut = 17;
constexpr uint32_t kCbOwn = 4;  // a share's chunk of this island's partial sums
constexpr uint32_t kCbFar = 5;  // the same chunk from each other island, in order
constexpr uint32_t kCbToken = 6;
constexpr uint32_t kCbArrive = 7;
constexpr uint32_t kBf16Bytes = 2048;
constexpr uint32_t kPartBytes = 4096;  // fp32
constexpr uint32_t kSignalBytes = 32;
constexpr uint32_t kChunk = 4;  // tiles of a share read (and asked) at once
constexpr uint32_t kBroadcastRow = 0;
constexpr uint32_t kBroadcastCol = 1;

// Ports: X's asked pages, P's asked pages; the island's tree up and down, the leaders'
// crossing, and the writer's go to its own reader.
constexpr uint32_t kPortX = 0;
constexpr uint32_t kPortP = 1;
constexpr uint32_t kPortUp = 2;
constexpr uint32_t kPortCross = 3;
constexpr uint32_t kPortDown = 4;
constexpr uint32_t kPortGo = 5;

// Arguments as the grid form's: reader, writer: broadcast words, then core words;
// compute: core words, then the two broadcast windows.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgCore = km::kBroadcastArgs;
constexpr uint32_t kCoreRank = 0;
constexpr uint32_t kCoreM0 = 1;
constexpr uint32_t kCoreN0 = 2;
constexpr uint32_t kCorePlan = 3;
constexpr uint32_t kPlMt = 0;
constexpr uint32_t kPlKt = 1;
constexpr uint32_t kPlNt = 2;
constexpr uint32_t kPlPcM = 3;
constexpr uint32_t kPlPcN = 4;
constexpr uint32_t kPlBm = 5;
constexpr uint32_t kPlBn = 6;
constexpr uint32_t kPlKb = 7;
constexpr uint32_t kPlNbh = 10;
constexpr uint32_t kPlNbw = 11;
constexpr uint32_t kPlSh = 12;
constexpr uint32_t kPlSw = 13;
constexpr uint32_t kPlanWords = 14;
constexpr uint32_t kCoreWords = kCorePlan + kPlanWords;

// A run of a tensor's pages: the first and the count.
struct Span {
    uint32_t first;
    uint32_t pages;
};

// The pages of a placement that island `at` holds (one island: all of them).
template <uint32_t Base, uint32_t I = 0>
inline Span held(uint32_t at) {
    if constexpr (I >= km::Placement<Base>::kIslands) {
        return {0, 0};
    } else {
        if (I == at) {
            return {
                get_compile_time_arg_val(Base + 2 + 2 * I),
                get_compile_time_arg_val(Base + 3 + 2 * I)
            };
        }
        return held<Base, I + 1>(at);
    }
}

// The K tiles this island's part of W covers: whole rows of W's [K, N] tiles.
struct Cover {
    uint32_t k0;
    uint32_t k_tiles;
};

template <uint32_t WBase>
inline Cover cover(uint32_t n_tiles) {
    const Span part = held<WBase>(km::island());
    return {part.first / n_tiles, part.pages / n_tiles};
}

inline uint32_t blocks_of(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}
