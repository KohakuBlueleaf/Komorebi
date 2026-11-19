// What the probe's kernels and Python agree on: the handler and the buffer in L1.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerProbe = 1;
constexpr uint32_t kProbeBuffer = 0;  // two 64-byte lines: written, then read back
