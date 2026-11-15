// Collectives across islands: the program's handlers.
#pragma once

#include <stdint.h>

// NOP: nothing (a command's own time, the baseline of the time a collective adds).
constexpr uint32_t kHandlerNop = 1;
// GATHER (reader): ranks 0 .. a4 - 1 post their share of pages a1 .. + a2 of tensor a0
// to island a3 (into a6), then wait; a8 kModeRing: from L1. a5 reports, a7 busy loop.
constexpr uint32_t kHandlerGather = 2;
constexpr uint32_t kModeGather = 0;
constexpr uint32_t kModeRing = 1;
constexpr uint32_t kPiece = 4;
// REDUCE (reader): two islands' partials (a0) of a4 kept pages each (a8, a9), staged in
// a1/a2, the sums into a3; a5 ranks | other << 16; a6 reports, a7 the busy loop.
constexpr uint32_t kHandlerReduce = 3;
constexpr uint32_t kPortKept = 0;
constexpr uint32_t kPortFolded = 1;
constexpr uint32_t kFold = 8;  // pages a fold step
// STAMPS (reader, one rank): the lead page CPU's KM_PUT_STAMPS words of link core 0
// toward island a0, and this core's wall clock, into page 0 of tensor a1.
constexpr uint32_t kHandlerStamps = 4;
// FILL (reader): pages a1 .. + a2 of a0 get the pattern of salt a3; CHECK: compared
// with it (a4: plus salt a4's, a sum), bad words into page `rank` of a5.
constexpr uint32_t kHandlerFill = 5;
constexpr uint32_t kHandlerCheck = 6;
constexpr uint32_t kPagePrime = 0x9E3779B1u;
constexpr uint32_t kWordPrime = 0x85EBCA77u;
constexpr uint32_t kCheckMagic = 0x43484B21u;  // "CHK!"
constexpr uint32_t kCheckBad = 2;
constexpr uint32_t kCheckFirstBad = 3;
constexpr uint32_t kCheckPages = 4;
// A rank's report: magic, rank, the wall clock at the start, after posting, after the
// last wait (or, busy, when the gather was seen done), at the end; the pages posted.
constexpr uint32_t kReportMagic = 0x434F4C4Cu;  // "COLL"
constexpr uint32_t kReportStart = 2;
constexpr uint32_t kReportPosted = 3;
constexpr uint32_t kReportDone = 4;
constexpr uint32_t kReportEnd = 5;
constexpr uint32_t kReportPages = 6;
constexpr uint32_t kReportFolded = 7;  // REDUCE: the wall clock when its fold ended
constexpr uint32_t kReportWords = 8;
// The compute's busy loop: a7 rounds of kBusyTiles matrix-unit products into the
// destination registers (no pack), on every rank's compute.
constexpr uint32_t kBusyTiles = 8;
// Buffers (circular buffer indices; their addresses only): a call's pages, the fold's
// three, the matrix unit's operands and output.
constexpr uint32_t kCbPages = 0;
constexpr uint32_t kCbA = 1;
constexpr uint32_t kCbB = 2;
constexpr uint32_t kCbOut = 16;
