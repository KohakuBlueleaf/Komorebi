// What the kernels and Python agree on: the handler, and the number of the buffer in L1
// in which the reader puts the tile.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerFetch = 1;
constexpr uint32_t kTileBuffer = 0;
constexpr uint32_t kPort = 0;  // where a tile from another card is counted
