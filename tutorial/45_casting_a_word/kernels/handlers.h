// What the kernels and Python agree on: the handler, the two choices the leader casts,
// and the buffers in L1 the reader uses: the values it reads and the total it writes.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerFollow = 1;
constexpr uint32_t kDouble = 1;    // the leader's word: double the total
constexpr uint32_t kAddThree = 2;  // the leader's word: add three to it
constexpr uint32_t kValues = 0;
constexpr uint32_t kTotal = 1;
