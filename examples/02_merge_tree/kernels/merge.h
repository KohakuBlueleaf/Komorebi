// Merge-tree example: the segment, buffers and record arguments its kernels share.
#pragma once

#include <stdint.h>

// MERGE: a0 input, a1 tiles T from tile a6, a2 output, a3 output page, a4 form (0
// owner, 1 tree), a5 member that stalls instead of sending (a test; none: 0xFFFFFFFF).
constexpr uint32_t kSegMerge = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbArrive = 1;  // the arrival buffer: slot k is page k
constexpr uint32_t kCbOut = 16;    // a member's part: sent, or written by member 0
constexpr uint32_t kInPages = 16;
constexpr uint32_t kArrivePages = 118;  // by owner: a slot for every other member
constexpr uint32_t kBlock = 8;          // input tiles per read barrier
constexpr uint32_t kTileBytes = 4096;   // float32 tiles everywhere
