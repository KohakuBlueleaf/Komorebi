// What the kernels and Python agree on: the handler, and the two buffers in L1: the one
// that holds the tile on its way out of the sending core, and the arrival buffer in
// which the receiving core's port puts what arrives.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerSend = 1;
constexpr uint32_t kOutgoing = 0;
constexpr uint32_t kArrival = 1;
