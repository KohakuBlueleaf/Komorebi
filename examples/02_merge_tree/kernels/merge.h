// Merge-tree example: the handler, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// MERGE: a0 input_address, a1 tiles (T), a2 output_address, a3 first_page, a4 form (0
// owner, 1 tree), a5 stalled_rank (none: 0xFFFFFFFF), a6 first_tile, a7 part_tiles.
constexpr uint32_t kHandlerMerge = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbArrive = 1;  // the arrival buffer: slot k is pages k * P on
constexpr uint32_t kCbOut = 16;    // a core's part: sent, or written by rank 0
constexpr uint32_t kInPages = 16;
constexpr uint32_t kArrivePages = 118;  // by owner: S / P slots, one per other rank
constexpr uint32_t kOutPages = 6;       // a part of P tiles never wraps (P <= 3)
constexpr uint32_t kPartMax = 3;        // P sums and one scratch tile in DST (fp32)
constexpr uint32_t kBlock = 8;          // input tiles per read barrier
constexpr uint32_t kTileBytes = 4096;   // float32 tiles everywhere
// Pages another island holds: the input's asked for at kAskPort, the total's put at
// kPutPort; rank 0 also counts the reduce's last rounds there (README).
constexpr uint32_t kAskPort = 6;
constexpr uint32_t kPutPort = 7;
