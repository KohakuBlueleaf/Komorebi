// The handlers of this program: the numbers the kernels and the host share.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerAdd = 1;  // result: a0 + a1
constexpr uint32_t kHandlerDot = 2;  // result: a0 * a3 + a1 * a4 + a2 * a5
constexpr uint32_t kHandlerWho = 3;  // result: the index of the core that ran it, plus a0
