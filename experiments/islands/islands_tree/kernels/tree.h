// The tree built-ins' experiment: the handler, buffers and command words its kernels
// share.
#pragma once

#include <stdint.h>

// ALL_REDUCE: a0 input (rank r's P tiles from page r * P), a1 output (rank r's sum at page
// r * P), a2 form (0 owner, 1 tree), a3 P tiles, a4 1: cast (every rank writes), 0: the
// plain reduce (rank 0 writes).
constexpr uint32_t kHandlerAllReduce = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbArrive = 1;  // the arrival buffer: slot k is pages k * P on
constexpr uint32_t kCbOut = 16;    // a core's part: sent; then its sum
constexpr uint32_t kInPages = 6;
constexpr uint32_t kArrivePages = 21;  // by tree: 7 rounds of up to 3 tiles
constexpr uint32_t kOutPages = 6;      // a part of P tiles never wraps (P <= 3)
constexpr uint32_t kPartMax = 3;       // P sums and one scratch tile in DST (fp32)
constexpr uint32_t kTileBytes = 4096;  // float32 tiles everywhere
