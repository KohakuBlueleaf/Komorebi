// Linear layer Y = X x W for islands: X, W split along K, Y along N; an island puts the
// columns others keep, then makes its own with theirs added. Words, buffers, ports.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_placement.hpp"

// LINEAR, reference words a0 X ([M, K] tiles), a1 W ([K, N]), a2 Y ([M, N]), a3 S (a
// copy: tiles put here, below), a6 T (0: none; a 64 B row a rank); a4 K, a5 N tiles.
constexpr uint32_t kHandlerLinear = 1;

// Buffers: the grid plan's (row and column broadcasts, bf16 output subblocks, fp32
// partial sums), then a block of the columns put across, the subblocks put here.
constexpr uint32_t kCbRow = 0;
constexpr uint32_t kCbCol = 1;
constexpr uint32_t kCbYOut = 16;
constexpr uint32_t kCbPart = 24;
constexpr uint32_t kCbSend = 17;  // bf16, one output block (pages consecutive)
constexpr uint32_t kCbSent = 5;   // bf16, an output subblock from one other island
constexpr uint32_t kCbToken = 6;
constexpr uint32_t kCbArrive = 7;
constexpr uint32_t kCbStamp = 8;  // the writer's row of T
constexpr uint32_t kBf16Bytes = 2048;
constexpr uint32_t kPartBytes = 4096;  // fp32
constexpr uint32_t kSignalBytes = 32;
constexpr uint32_t kStampBytes = 64;
// T's words: the command taken, the other islands' columns put (sent), the sync done,
// its own columns written (the wall clock, AICLK cycles).
constexpr uint32_t kStampStart = 0;
constexpr uint32_t kStampPut = 1;
constexpr uint32_t kStampSynced = 2;
constexpr uint32_t kStampEnd = 3;
constexpr uint32_t kBroadcastRow = 0;
constexpr uint32_t kBroadcastCol = 1;

// Ports: the writer's puts counted; the island's tree up and down, the leaders'
// crossing, the writer's go to its own reader.
constexpr uint32_t kPortPut = 1;
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

// What island `at` holds of a tensor (its reference word, `pages` pages): its part's
// bank address, steps first .. of `all` steps of `inner` pages, in `blocks` blocks.
struct Holding {
    uint32_t address;
    uint32_t first;
    uint32_t steps;
    uint32_t all;
    uint32_t inner;
    uint32_t pages;  // the part's

    // Page `index` of the part in the whole tensor, and back.
    uint32_t page(uint32_t index) const {
        const uint32_t block = index / (steps * inner), rest = index % (steps * inner);
        return (block * all + first) * inner + rest;
    }

    uint32_t index(uint32_t page) const {
        const uint32_t block = page / (all * inner), step = page / inner % all;
        return (block * steps + step - first) * inner + page % inner;
    }

    // Whether pages page .. page + count - 1 are all of the part, so at consecutive
    // indices from index(page).
    bool holds(uint32_t page, uint32_t count) const {
        const uint32_t last = page + count - 1, span = all * inner;
        return (
            page / span == last / span && page / inner % all - first < steps &&
            last / inner % all - first < steps
        );
    }
};

inline Holding holding(uint32_t tensor, uint32_t at, uint32_t pages) {
    const km::Islands view(km::island_area());
    const volatile uint32_t* entry = km::tensor_entry(view, tensor);
    if (entry == nullptr) {  // a bank address: one island holds every page
        return {tensor, 0, 1, 1, pages, pages};
    }
    const volatile uint32_t*
        part = (entry + KM_TENSOR_HEADER_BYTES / 4 + at * km::kPartWords);
    if (entry[3] & 1u) {  // a copy: every island all of it
        return {part[0], 0, 1, 1, pages, pages};
    }
    const uint32_t all = entry[1], inner = entry[2];
    return {
        part[0], part[1], part[2], all, inner, pages / (all * inner) * part[2] * inner
    };
}

// Tiles first .. first + count - 1 of a dimension.
struct Span {
    uint32_t first;
    uint32_t count;
};

// The K tiles of W island `at` holds: whole rows of W ([K, N] tiles, split along K),
// or all of it.
inline Span k_part(uint32_t w, uint32_t at, uint32_t k_whole, uint32_t n_whole) {
    const Holding part = holding(w, at, k_whole * n_whole);
    if (part.pages == k_whole * n_whole) {
        return {0, k_whole};
    }
    if (part.pages != 0 && part.all == k_whole && part.inner == n_whole) {
        return {part.first, part.steps};
    }
    return {0, 0};
}

// The columns of Y ([M, N] tiles, split along N, or all of it) island `at` keeps.
inline Span columns(uint32_t y, uint32_t at, uint32_t m_tiles, uint32_t n_whole) {
    const Holding part = holding(y, at, m_tiles * n_whole);
    if (part.pages == m_tiles * n_whole) {
        return {0, n_whole};
    }
    if (part.pages != 0 && part.all == n_whole && part.inner == 1) {
        return {part.first, part.steps};
    }
    return {0, 0};
}

// The islands in the order an island computes their columns: the next one first, its
// own last (`turn` 1 .. islands).
inline uint32_t island_at(uint32_t here, uint32_t turn) {
    return (here + turn) % km::islands();
}

// Whether any island puts to another: one holds part of K and another keeps columns.
// The same answer on every island (no sync when false).
inline bool exchanging(uint32_t w, uint32_t y, uint32_t m, uint32_t k, uint32_t n) {
    for (uint32_t from = 0; from < km::islands(); ++from) {
        if (k_part(w, from, k, n).count == 0) {
            continue;
        }
        for (uint32_t to = 0; to < km::islands(); ++to) {
            if (to != from && columns(y, to, m, n).count != 0) {
                return true;
            }
        }
    }
    return false;
}

inline uint32_t blocks_of(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// S: a set a sending island (sent_set), in a set each plan core's output blocks in the
// order it makes them (core m0 / pcm x blocks_of(nt, pcn) + n0 / pcn), tiles as packed.
struct Sent {
    uint32_t cores;
    uint32_t blocks;
    uint32_t block_tiles;

    uint32_t page(uint32_t set, uint32_t core, uint32_t block) const {
        return ((set * cores + core) * blocks + block) * block_tiles;
    }
};

inline uint32_t sent_set(uint32_t from, uint32_t to) {
    return (from + km::islands() - to - 1) % km::islands();
}
