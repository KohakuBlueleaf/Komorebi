// What the kernels and Python agree on: the dynamic handler, and the words of a
// message.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerWander = 1;

// A message is 16 words of 4 bytes; word 0 belongs to Komorebi.
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kHopsLeft = 1;
constexpr uint32_t kValue = 2;
