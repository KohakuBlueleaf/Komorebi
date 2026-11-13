// What the kernels and Python agree on: the handler, and the number of the buffer in L1
// in which the tile waits between the reader and the writer.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerCopy = 1;
constexpr uint32_t kTileBuffer = 0;
