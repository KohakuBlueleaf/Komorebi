// What the kernels and Python agree on: the handler, and three buffers in L1: the
// core's own tiles, the arrival buffer for other cores' sums, and the output.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerSumTiles = 1;
constexpr uint32_t kOwnTiles = 0;
constexpr uint32_t kArrival = 1;
constexpr uint32_t kArrivalPages = 8;
constexpr uint32_t kOutput = 16;
