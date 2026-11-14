// Pages across islands: the program's handlers.
#pragma once

#include <stdint.h>

// WHERE: the writer's result is km::island_of(tensor a0, page a1) on this core (a0:
// the tensor's reference word).
constexpr uint32_t kHandlerWhere = 1;
// MOVE (reader): the rank's share of pages 0 .. a2 - 1 of tensor a0, a3 a call:
// km::read_pages into L1, then km::write_pages into a1; another island's pages cross.
constexpr uint32_t kHandlerMove = 2;
// SOAK (reader): MOVE's calls, a4 rounds, a3 pages a call; asks mark each page in L1
// and check it after; puts: km::write_pages into a1. Report: page `rank` of a5.
constexpr uint32_t kHandlerSoak = 3;
constexpr uint32_t kSoakMagic = 0x534F414Bu;  // "SOAK": word 0 of a report
constexpr uint32_t kSoakMark = 0xDEAD0000u;   // a page's word 0 before its ask
constexpr uint32_t kSoakNone = 0xFFFFFFFFu;
// report words: magic, calls done, bad pages (asks), first bad page, missing page of
// the call that gave up, its round (kSoakNone: none), the rank
constexpr uint32_t kSoakCalls = 1;
constexpr uint32_t kSoakBad = 2;
constexpr uint32_t kSoakFirstBad = 3;
constexpr uint32_t kSoakMissing = 4;
constexpr uint32_t kSoakFailed = 5;
constexpr uint32_t kSoakRank = 6;
constexpr uint32_t kSoakWords = 8;
// RUNS (reader): the rank's share of a0 put into a1 (another island's) in runs of a3
// pages (km::put_pages, km::put_piece a4 pages), one km::put_wait. With a5: a report.
constexpr uint32_t kHandlerRuns = 4;
constexpr uint32_t kRunsMagic = 0x52554E53u;  // "RUNS": word 0 of a report
constexpr uint32_t kRunsLine = 0;             // km::put_pages: the run's line
constexpr uint32_t kRunsRead = 1;             // the piece read from a0 into L1
constexpr uint32_t kRunsPiece = 2;            // km::put_piece
constexpr uint32_t kRunsWait = 3;             // the last km::put_wait
constexpr uint32_t kRunsSteps = 4;
// The L1 buffer (a circular buffer index; only its address is used).
constexpr uint32_t kCbMove = 0;
