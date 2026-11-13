// Micro-op stream example: the handler, buffers and command arguments its kernels
// share.
#pragma once

#include <stdint.h>

// MAP: a0 input_address, a1 tiles (T), a2 output_address, a3 first_page (of rank 0), a4
// first_tile.
constexpr uint32_t kHandlerMap = 1;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
