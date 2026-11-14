// Pages across islands: the program's handlers.
#pragma once

#include <stdint.h>

// WHERE: the writer's result is km::island_of(tensor a0, page a1) on this core (a0:
// the tensor's reference word).
constexpr uint32_t kHandlerWhere = 1;
// MOVE (reader): the rank's share of pages 0 .. a2 - 1 of tensor a0 (its reference
// word), a3 pages a call: km::read_pages into its L1 buffer, then km::write_pages
// into the same pages of tensor a1. A tensor another island holds is asked for
// (read) or put (written) across; this island's pages move as tt-metal's page calls.
constexpr uint32_t kHandlerMove = 2;
// The L1 buffer (a circular buffer index; only its address is used).
constexpr uint32_t kCbMove = 0;
