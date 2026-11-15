// Matrix product with K split: a group of S cores shares one output block, each core
// takes 1/S of K, and the parts are summed by owner or by tree over the cores' ports.
#pragma once

#include <stdint.h>

#include "../dense.h"

// SPLIT: a0 A, a1 W, a2 Y, a3 reduce form (0 owner, 1 tree) | kSplitSecondSet, a4 mode,
// a5 the plan (0 the first), a6 the operation after the sum (split_op.hpp), a7-a9 its.
constexpr uint32_t kHandlerSplit = 1;

// a3: the sum in the upper half of ports and arrival slots (sets.hpp), in flight beside
// a command without it; every plan's sum then fits a half (KM_PORTS / 2 tree rounds).
constexpr uint32_t kSplitSecondSet = 0x100;

inline uint32_t split_form(const km_command_t& command) {
    return command.a3 & 0xFFu;
}

inline bool split_second_set(const km_command_t& command) {
    return (command.a3 & kSplitSecondSet) != 0;
}

// SPLIT modes: the full call; the parts only (no reduce: each core packs its part and
// drops it, nothing written); the operation only (no product: every core calls it).
constexpr uint32_t kSplitFull = 0;
constexpr uint32_t kSplitParts = 1;
constexpr uint32_t kSplitOp = 2;

// A and W chunks in two pairs (chunks.hpp): a half holds one chunk of its mover (mover
// A's half may hold more: the plan's depth, dense/plan.py).
constexpr uint32_t kCbSa = 0;        // mover A's A chunks: oh x kc tiles
constexpr uint32_t kCbSw = 1;        // mover A's W chunks: kc x ow tiles
constexpr uint32_t kCbSaMoverB = 5;  // mover B's A chunks
constexpr uint32_t kCbSwMoverB = 6;  // mover B's W chunks
constexpr uint32_t kCbArrive = 2;  // arrival buffer, fp32: a slot of P tiles per sender
constexpr uint32_t kCbSum = 24;    // this core's part, fp32: P = oh x ow tiles
constexpr uint32_t kCbSplitOut = 16;  // the group's sum in bf16 at its root: P tiles
constexpr uint32_t kSumBytes = 4096;

// The plan, the same words on every core's kernels: tiles M, K, N; group size S; block
// oh x ow; K chunk kc; subblock sh x sw; blocks per row; arrival pages; pattern.
constexpr uint32_t kSpMt = 0;
constexpr uint32_t kSpKt = 1;
constexpr uint32_t kSpNt = 2;
constexpr uint32_t kSpS = 3;
constexpr uint32_t kSpOh = 4;
constexpr uint32_t kSpOw = 5;
constexpr uint32_t kSpKc = 6;
constexpr uint32_t kSpSh = 7;
constexpr uint32_t kSpSw = 8;
constexpr uint32_t kSpNbw = 9;
constexpr uint32_t kSpArrive = 10;
constexpr uint32_t kSpPattern = 11;
constexpr uint32_t kSplitWords = 12;
constexpr uint32_t kSplitAB = 0;
constexpr uint32_t kSplitAAAB = 1;
// A program takes up to kSplitPlans plans, plan p at argument p * kSplitWords: shapes
// that share every buffer (oh, ow, kc, sh, sw and the arrival pages).
constexpr uint32_t kSplitPlans = 8;

// The first argument of the plan a command names (its a5).
inline uint32_t plan_words(const km_command_t& command) {
    return command.a5 * kSplitWords;
}
