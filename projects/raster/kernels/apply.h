// APPLY: y = M x of a light-transport set on every core (projects/transport's handler;
// README.md "APPLY"): its command words, sizes, table words and buffers.
#pragma once

#include <stdint.h>

#include "komorebi/device/movement/km_movement.hpp"
#if !defined(COMPILE_FOR_TRISC)
#include "komorebi/device/tensors/km_ask.hpp"
#endif

// APPLY: a0 x (bf16 tiles), a1 w (bfp8_b pages), a2 y (float32 tiles), a3 tables, a4
// profile (a page a core: 4 unpack words, then 4 pack words), a5 unused.
constexpr uint32_t kHandlerApply = 7;

// The set's sizes (apply.py, from plan.sizes at load): own receiver tiles, x's tiles of
// an entry, rank tiles, table pages a core, weight tiles a block.
#ifndef APPLY_OWN
#define APPLY_OWN 16
#define APPLY_KMAX 72
#define APPLY_RANK 5
#define APPLY_PAGES 4
#define APPLY_BLOCK 8
#endif
constexpr uint32_t kApplyOwn = APPLY_OWN;
constexpr uint32_t kApplyKmax = APPLY_KMAX;
constexpr uint32_t kApplyRank = APPLY_RANK;
constexpr uint32_t kApplyPages = APPLY_PAGES;
constexpr uint32_t kApplyBlock = APPLY_BLOCK;
// Pack's own-tile mask: a bit an own sums tile, in as many words as kApplyOwn needs.
constexpr uint32_t kApplyOwnWords = (kApplyOwn + 31) / 32;

// A table page (plan.py): word 0 its entries, 1 and 2 the core's first own tile and
// their count; entries of 8 words from word 8, at most 127 a page.
constexpr uint32_t kApplyHead = 8;
constexpr uint32_t kApplyEntryWords = 8;
constexpr uint32_t kApplyEntriesMost = 127;
constexpr uint32_t kApplyHeadFirst = 1;
constexpr uint32_t kApplyHeadOwn = 2;
// An entry's words (transport.Slice): x's tiles src : src + k, the first stream's pages
// w : w + a, U^T's u : u + b, the first own receiver r, R tiles of rank (0: dense).
constexpr uint32_t kApplySrc = 0;
constexpr uint32_t kApplyK = 1;
constexpr uint32_t kApplyW = 2;
constexpr uint32_t kApplyA = 3;
constexpr uint32_t kApplyU = 4;
constexpr uint32_t kApplyB = 5;
constexpr uint32_t kApplyR = 6;
constexpr uint32_t kApplyRankWord = 7;

// Page sizes: x bf16, w bfp8_b, y float32, a table and a profile page.
constexpr uint32_t kApplyXBytes = 2048;
constexpr uint32_t kApplyWBytes = 1088;
constexpr uint32_t kApplyYBytes = 4096;
constexpr uint32_t kApplyTableBytes = 4096;
constexpr uint32_t kApplyProfBytes = 4096;

// The read halves (below 32, not RASTER_SPREAD's 10 to 12 nor 29 to 31): x's held
// block, the weights (an entry's V, then its U^T: the stream armed again).
constexpr uint32_t kCbApplyX = 25;
constexpr uint32_t kCbApplyXB = 26;
constexpr uint32_t kCbApplyW = 27;
constexpr uint32_t kCbApplyWB = 28;
// y's tiles and the profile page: pack fills them, mover B writes them (no stream).
constexpr uint32_t kCbApplyY = 32;
constexpr uint32_t kCbApplyProf = 33;
constexpr uint32_t kApplyYPages = 2;
// The table pages: mover A's to compute, mover B's its own copy; the head page the
// same.
constexpr uint32_t kCbApplyEntries = 34;
constexpr uint32_t kCbApplyEntriesB = 35;
constexpr uint32_t kCbApplyHeadPage = 36;
constexpr uint32_t kCbApplyHeadPageB = 37;
// Compute's own: the sums (float32, kApplyOwn tiles), t (bf16, kApplyRank tiles, t^T),
// the token pack hands unpack (bf16, 2 tiles).
constexpr uint32_t kCbApplySums = 38;
constexpr uint32_t kCbApplyHidden = 39;
constexpr uint32_t kCbApplyReady = 40;
constexpr uint32_t kApplyReadyPages = 2;
constexpr uint32_t kApplySumsBytes = 4096;
constexpr uint32_t kApplyHiddenBytes = 2048;
// The own program's host buffer of the windows (apply.py --windows): plain memory.
constexpr uint32_t kCbApplyHost = 41;

constexpr km::BufferPair kApplyXPair = {kCbApplyX, kCbApplyXB};
constexpr km::BufferPair kApplyWPair = {kCbApplyW, kCbApplyWB};
// The reads' pattern: mover B every fourth block.
constexpr km::Pattern kApplyReadPattern = km::pattern("AAAB");

#if !defined(COMPILE_FOR_TRISC)
// The five tensors (km::Tensor, compile arguments from `At` on): x, w, y, the tables,
// the profile pages.
template <uint32_t At>
struct ApplyTensors {
    using X = km::Tensor<At>;
    using W = km::Tensor<X::next_compile_time_args_offset()>;
    using Y = km::Tensor<W::next_compile_time_args_offset()>;
    using Tables = km::Tensor<Y::next_compile_time_args_offset()>;
    using Profile = km::Tensor<Tables::next_compile_time_args_offset()>;
};
#endif
