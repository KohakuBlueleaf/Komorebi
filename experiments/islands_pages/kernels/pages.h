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
// SOAK (reader): MOVE's calls a4 rounds over the rank's share of pages 0 .. a2 - 1 of
// tensor a0, a3 pages a call. From another island (asks): before each call a mark in
// word 0 of each page's L1, after it word 0 must be the page's index and the last
// word its complement (the host's pattern), no copy made; from this island (puts):
// km::write_pages into tensor a1. Then the rank's report into page `rank` of tensor
// a5 (this island's): kSoak* words; on a call that gave up, the first page whose mark
// is still there named (asks), and the report written all the same.
constexpr uint32_t kHandlerSoak = 3;
constexpr uint32_t kSoakMagic = 0x534F414Bu;  // "SOAK": word 0 of a report
constexpr uint32_t kSoakMark = 0xDEAD0000u;   // a page's word 0 before its ask
constexpr uint32_t kSoakNone = 0xFFFFFFFFu;
// a report's words: magic, calls done, pages bad (asks), first bad page, the missing
// page of the call that gave up (kSoakNone: none or a put), the round it gave up in
// (kSoakNone: none), the rank
constexpr uint32_t kSoakCalls = 1;
constexpr uint32_t kSoakBad = 2;
constexpr uint32_t kSoakFirstBad = 3;
constexpr uint32_t kSoakMissing = 4;
constexpr uint32_t kSoakFailed = 5;
constexpr uint32_t kSoakRank = 6;
constexpr uint32_t kSoakWords = 8;
// RUNS (reader): the rank's share of pages 0 .. a2 - 1 of tensor a0 (this island's)
// put into the same pages of tensor a1 (another island's) in runs of a3 pages, each
// named by km::put_pages and its bytes following by km::put_piece, a4 pages a piece
// (read from a0 into L1 just before), then one km::put_wait for all its runs.
constexpr uint32_t kHandlerRuns = 4;
// The L1 buffer (a circular buffer index; only its address is used).
constexpr uint32_t kCbMove = 0;
