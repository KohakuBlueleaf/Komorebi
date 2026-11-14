// Island-aware sum: the handler, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// SUM: a0 input_address, a1 tiles (T), a2 output_address, a3 first_page, a4 form (0
// owner, 1 tree) inside an island, a6 first_tile, a7 part_tiles.
constexpr uint32_t kHandlerSum = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbArrive = 1;  // the arrival buffer: slot k is pages k * P on
constexpr uint32_t kCbOut = 16;    // a core's part: sent, or written by the root
constexpr uint32_t kInPages = 16;
constexpr uint32_t kArrivePages = 118;
constexpr uint32_t kOutPages = 6;      // a part of P tiles never wraps (P <= 3)
constexpr uint32_t kPartMax = 3;       // P sums and one scratch tile in DST (fp32)
constexpr uint32_t kBlock = 8;         // input tiles per read barrier
constexpr uint32_t kTileBytes = 4096;  // float32 tiles everywhere
constexpr uint32_t kCrossPort = 7;     // the leaders' parts at the first leader

// The arrival slot of the part of island part j > 0 at the first leader: from the end.
inline uint32_t cross_slot(uint32_t part_tiles, uint32_t j) {
    return kArrivePages / part_tiles - j;
}
