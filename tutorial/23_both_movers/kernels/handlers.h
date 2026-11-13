// What the kernels and Python agree on: the handler, the two halves of the buffer pair
// (one for each data-movement CPU), and how many tiles make a block.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerCopy = 1;
constexpr uint32_t kHalfA = 0;
constexpr uint32_t kHalfB = 1;
constexpr uint32_t kBlockTiles = 8;
