// The handlers of this program: the numbers the kernels and the host share.
#pragma once

#include <stdint.h>

constexpr uint32_t kSegAdd = 1;  // result: a0 + a1
constexpr uint32_t kSegDot = 2;  // result: a0 * a3 + a1 * a4 + a2 * a5
constexpr uint32_t kSegWho = 3;  // result: the index of the core that ran it, plus a0
