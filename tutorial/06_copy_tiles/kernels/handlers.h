// The handlers of this program, its buffer and the page size the kernels share.
#pragma once

#include <stdint.h>

// COPY: a0 input tensor address, a1 output tensor address, a2 number of tiles.
constexpr uint32_t kSegCopy = 1;

constexpr uint32_t kCb = 0;            // the circular buffer between reader and writer
constexpr uint32_t kTileBytes = 2048;  // one bfloat16 tile: 32 x 32 x 2 bytes
