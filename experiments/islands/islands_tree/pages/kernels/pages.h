// The tree collectives on a copy's pages: the handler and the buffer its kernels share.
#pragma once

#include <stdint.h>

// TREE: a0 the copy's reference word, a1 first page, a2 pages, a3 the scratch's, a4
// km::Sum, a5 the call (kCall*), a6 the root island.
constexpr uint32_t kHandlerTree = 1;
constexpr uint32_t kCallGather = 0;
constexpr uint32_t kCallReduce = 1;
constexpr uint32_t kCallReduceTo = 2;
constexpr uint32_t kCallBroadcast = 3;

constexpr uint32_t kCbFold = 0;     // the folds' L1: pages read, added, written
constexpr uint32_t kFoldPages = 8;  // of the largest page the experiment uses
constexpr uint32_t kFoldBytes = 4096;
