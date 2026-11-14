// All-reduce across islands: the handler and the buffer its kernels share.
#pragma once

#include <stdint.h>

// REDUCE: a0 the tensor's reference word (a copy), a1 first page, a2 pages, a3 the
// scratch's (a copy of 2 x pages pages), a4 what is added (km::Sum).
constexpr uint32_t kHandlerReduce = 1;

constexpr uint32_t kCbFold = 0;      // the folds' L1: pages read, added, written
constexpr uint32_t kFoldPages = 8;   // of the largest page the example uses
constexpr uint32_t kFoldBytes = 4096;
