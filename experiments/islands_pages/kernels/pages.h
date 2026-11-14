// Pages across islands: the program's handlers and the tensors its kernels know.
#pragma once

#include <stdint.h>

// WHERE: the writer's result is km::island_of(tensor a0, page a1) on this core.
constexpr uint32_t kHandlerWhere = 1;
// ASK (reader): the rank's share of pages 0 .. a2 - 1 of the tensor at a0 (the
// reader's first placement, another island holds it), a3 pages a call of
// km::read_pages into its L1 buffer, each call's pages then written into the same
// pages of the tensor at a1 (this island's).
constexpr uint32_t kHandlerAsk = 2;
// PUT (writer): the rank's share of pages 0 .. a2 - 1 of the tensor at a0 (this
// island's), a3 pages at a time read into its L1 buffer and written with
// km::write_pages into the same pages of the tensor at a1 (the writer's second
// placement, another island holds it).
constexpr uint32_t kHandlerPut = 3;
// The L1 buffers (circular buffer indices; only their addresses are used) and ports.
constexpr uint32_t kCbAsk = 0;
constexpr uint32_t kCbPut = 1;
constexpr uint32_t kAskPort = 2;
constexpr uint32_t kPutPort = 3;
