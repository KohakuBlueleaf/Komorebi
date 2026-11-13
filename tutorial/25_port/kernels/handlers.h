// What the kernels and Python agree on: the handler, and two buffers in L1: the
// outgoing tile of the sender, and the arrival buffer of the receiver.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerSend = 1;
constexpr uint32_t kOutgoing = 0;
constexpr uint32_t kArrival = 1;
