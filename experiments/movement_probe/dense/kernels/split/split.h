// The K split (a copy of the program's) with its K chunks split over both movers: chunk
// q of a core is read by the mover CHUNK_PATTERN names, into that mover's halves.
#pragma once

#include <stdint.h>

#include "../dense.h"

// SPLIT: a0 A ([M, K] tiles), a1 W ([K, N]), a2 Y ([M, N]), a3 reduce form (0 owner,
// 1 tree), a4 mode. One call; the group is the groups' cores in order, S to a group.
constexpr uint32_t kHandlerSplit = 1;

// SPLIT modes: the full call; the parts only (no reduce: each core packs its part and
// drops it, nothing written).
constexpr uint32_t kSplitFull = 0;
constexpr uint32_t kSplitParts = 1;

constexpr uint32_t kCbSa = 0;  // mover A's A chunks: one of oh x kc tiles
constexpr uint32_t kCbSw = 1;  // mover A's W chunks: one of kc x ow tiles
// Mover B's halves of the A and W chunk buffers (kCbSa and kCbSw are mover A's).
constexpr uint32_t kChunkAMoverB = 5;
constexpr uint32_t kChunkWMoverB = 6;
#if !defined(CHUNK_PATTERN)
#define CHUNK_PATTERN 0x001
#endif
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
