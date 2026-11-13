// The handlers of this program, its buffer and the page size the kernels share.
#pragma once

#include <stdint.h>

// SUM: a0 input tensor address, a1 number of pages (rows). Result: the sum of all
// words.
constexpr uint32_t kHandlerSum = 1;

constexpr uint32_t kCb = 0;
constexpr uint32_t kPageWords = 256;  // one row of the input: 256 words of 4 bytes
