// The handlers of this program: the numbers the kernels and the host share.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerAdd = 1;  // a0 + a1, returned as the result word
