// What the kernels and Python agree on: the handler, and the number of the buffer in L1
// in which the tile waits between the reader and the writer.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerCopy = 1;
constexpr uint32_t kTileBuffer = 0;

// The value of a5 that switches a bug on: the reader leaves out cb_push_back, or the
// writer leaves out cb_pop_front. Any other value: no bug.
constexpr uint32_t kBugForgetPush = 1;
constexpr uint32_t kBugForgetPop = 2;
