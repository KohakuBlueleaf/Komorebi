// Matrix product with K split: a group of S cores shares one output block, each core
// takes 1/S of K, and the parts are summed by owner or by tree over the cores' ports.
#pragma once

#include <stdint.h>

#include "../dense.h"

// SPLIT: a0 A, a1 W, a2 Y (tile tensors), a3 reduce_form (0 owner, 1 tree), a4 mode, a5
// the plan (0 the first), a6 the operation after the sum (split_op.hpp), a7 to a9 its.
constexpr uint32_t kHandlerSplit = 1;

// SPLIT modes: the full call; the parts only (no reduce: each core packs its part and
// drops it, nothing written); the operation only (no product: every core calls it).
constexpr uint32_t kSplitFull = 0;
constexpr uint32_t kSplitParts = 1;
constexpr uint32_t kSplitOp = 2;

// A and W chunks in two pairs (chunks.hpp): each half holds one chunk of its mover.
constexpr uint32_t kCbSa = 0;        // mover A's A chunks: oh x kc tiles
constexpr uint32_t kCbSw = 1;        // mover A's W chunks: kc x ow tiles
constexpr uint32_t kCbSaMoverB = 5;  // mover B's A chunks
constexpr uint32_t kCbSwMoverB = 6;  // mover B's W chunks
constexpr uint32_t kCbArrive = 2;  // arrival buffer, fp32: a slot of P tiles per sender
constexpr uint32_t kCbSum = 24;    // this core's part, fp32: P = oh x ow tiles
constexpr uint32_t kCbSplitOut = 16;  // the group's sum in bf16 at its root: P tiles
constexpr uint32_t kSumBytes = 4096;

// The plan, the same words on every kernel of a core: tiles M, K, N; group size S;
// block oh x ow; K chunk kc; subblock sh x sw; blocks per row; arrival buffer pages.
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
constexpr uint32_t kSplitWords = 11;
// A program takes up to kSplitPlans plans, plan p at argument p * kSplitWords: shapes
// that share every buffer (oh, ow, kc, sh, sw and the arrival pages).
constexpr uint32_t kSplitPlans = 8;

// The first argument of the plan a command names (its a5).
inline uint32_t plan_words(const km_command_t& command) {
    return command.a5 * kSplitWords;
}
