// What the kernels and Python agree on: the handler, and the three buffers in L1: the
// tiles of the core's own share, the arrival buffer for the sums of the other cores,
// and the output.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerSumTiles = 1;
constexpr uint32_t kOwnTiles = 0;
constexpr uint32_t kArrival = 1;
constexpr uint32_t kArrivalPages = 8;
constexpr uint32_t kOutput = 16;
