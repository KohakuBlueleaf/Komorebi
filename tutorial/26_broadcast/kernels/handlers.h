// What the kernels and Python agree on: the handler, and the buffers in L1: the
// broadcast buffer, in which every core of the row receives the blocks, and the output.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerBroadcast = 1;
constexpr uint32_t kRowBuffer = 0;
constexpr uint32_t kOutput = 16;
