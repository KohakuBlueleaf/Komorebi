// What the kernels and Python agree on: the handler, and the numbers of the three
// buffers in L1: the two inputs of the addition and its output.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerMap = 1;
constexpr uint32_t kFirstInput = 0;
constexpr uint32_t kSecondInput = 1;
constexpr uint32_t kOutput = 16;
