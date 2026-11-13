// Merge-tree example: the handler, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// MERGE: a0 input, a1 tiles T from tile a6, a2 output, a3 first output page, a4 form
// (0 owner, 1 tree), a5 member that stalls (a test; none: 0xFFFFFFFF), a7 part tiles P.
constexpr uint32_t kHandlerMerge = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbArrive = 1;  // the arrival buffer: slot k is pages k * P on
constexpr uint32_t kCbOut = 16;    // a member's part: sent, or written by member 0
constexpr uint32_t kInPages = 16;
constexpr uint32_t kArrivePages = 118;  // by owner: S / P slots, one per other member
constexpr uint32_t kOutPages = 6;       // a part of P tiles never wraps (P <= 3)
constexpr uint32_t kPartMax = 3;        // P sums and one scratch tile in DST (fp32)
constexpr uint32_t kBlock = 8;          // input tiles per read barrier
constexpr uint32_t kTileBytes = 4096;   // float32 tiles everywhere
