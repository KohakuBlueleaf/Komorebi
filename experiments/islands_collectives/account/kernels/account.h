// The collectives' account: the handlers and the fold buffer its kernels share.
#pragma once

#include <stdint.h>

// GATHER: a0 tensor (a copy), a1 first page, a2 pages; REDUCE: also a3 the scratch,
// a4 what is added (km::Sum).
constexpr uint32_t kHandlerGather = 1;
constexpr uint32_t kHandlerReduce = 2;

constexpr uint32_t kCbFold = 0;  // the folds' L1
constexpr uint32_t kFoldPages = 8;
constexpr uint32_t kFoldBytes = 4096;
