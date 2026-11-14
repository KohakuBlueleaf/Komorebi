// Linear layer Y = X x W written for islands: each island computes with the part of W
// it holds, the grid plan inside it. Handlers, buffers, arguments and ports follow.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_placement.hpp"

// LINEAR, the tensors' reference words: a0 X ([M, K] tiles), a1 W ([K, N]), a2 Y ([M,
// N]), a3 P ([M, N] fp32 on every island: its partial sums); a4 K, a5 N tiles.
constexpr uint32_t kHandlerLinear = 1;

// Buffers: the grid plan's (row and column broadcasts, bf16 output subblocks, fp32
// partial sums), then fp32 output subblocks of a partial product, and the shares.
constexpr uint32_t kCbRow = 0;
constexpr uint32_t kCbCol = 1;
constexpr uint32_t kCbYOut = 16;
constexpr uint32_t kCbPart = 24;
constexpr uint32_t kCbPartOut = 17;
constexpr uint32_t kCbSums = 5;  // a share's chunk of each holder's partial sums, in order
constexpr uint32_t kCbToken = 6;
constexpr uint32_t kCbArrive = 7;
constexpr uint32_t kBf16Bytes = 2048;
constexpr uint32_t kPartBytes = 4096;  // fp32
constexpr uint32_t kSignalBytes = 32;
constexpr uint32_t kChunk = 4;  // tiles of a share read (and asked) at once
constexpr uint32_t kBroadcastRow = 0;
constexpr uint32_t kBroadcastCol = 1;

// Ports: X's and P's asked pages; the island's tree up and down, the leaders' crossing,
// the writer's go to its own reader; Y's pages put across.
constexpr uint32_t kPortX = 0;
constexpr uint32_t kPortP = 1;
constexpr uint32_t kPortUp = 2;
constexpr uint32_t kPortCross = 3;
constexpr uint32_t kPortDown = 4;
constexpr uint32_t kPortGo = 5;
constexpr uint32_t kPortY = 6;

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

// The block of W's [K, N] tiles island `at` holds: K tiles k0 .., N tiles n0 ...
struct Cover {
    uint32_t k0;
    uint32_t k_tiles;
    uint32_t n0;
    uint32_t n_tiles;
};

inline Cover cover(uint32_t w, uint32_t at, uint32_t k_whole, uint32_t n_whole) {
    const Holding part = holding(w, at, k_whole * n_whole);
    if (part.pages == 0 || part.pages == k_whole * n_whole) {  // none, or all of W
        return {0, part.pages != 0 ? k_whole : 0, 0, part.pages != 0 ? n_whole : 0};
    }
    if (part.all * part.inner == k_whole * n_whole && part.inner == n_whole) {
        return {part.first, part.steps, 0, n_whole};  // whole rows: part of K
    }
    if (part.all == n_whole && part.inner == 1) {
        return {0, k_whole, part.first, part.steps};  // whole columns: part of N
    }
    return {0, 0, 0, 0};  // no block of W's tiles: nothing computed here
}

// The islands whose part of W holds some of K, a bit each; `whole`: each holds all of
// it (its product is its columns of Y), else each a partial sum of all of Y.
struct Parts {
    uint32_t holders;
    bool whole;
};

inline Parts parts(uint32_t w, uint32_t k_whole, uint32_t n_whole) {
    Parts found = {0, true};
    for (uint32_t at = 0; at < km::islands(); ++at) {
        const uint32_t k_tiles = cover(w, at, k_whole, n_whole).k_tiles;
        found.holders |= (k_tiles != 0 ? 1u : 0u) << at;
        found.whole = found.whole && (k_tiles == 0 || k_tiles == k_whole);
    }
    return found;
}

inline uint32_t blocks_of(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator - 1) / denominator;
}
