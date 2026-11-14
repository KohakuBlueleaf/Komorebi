// Collectives across islands: the program's handlers.
#pragma once

#include <stdint.h>

// NOP: nothing (a command's own time, the baseline of the time a collective adds).
constexpr uint32_t kHandlerNop = 1;
// GATHER (reader): ranks 0 .. a4 - 1 of the command each post their share of pages
// a1 .. a1 + a2 - 1 of tensor a0 (km::gather_pages) to island a3, into tensor a6 (0:
// a0, a copy), then wait once (km::gather_wait). Mode a8: kModeRing puts the share from
// L1 instead (km::put_pages, put_piece of kPiece pages, put_wait: the linear kernel's
// put). With a5 (a tensor of this island's) the rank's report into its page `rank`.
// a7: the compute's busy loop (kBusy*), every rank of the command.
constexpr uint32_t kHandlerGather = 2;
constexpr uint32_t kModeGather = 0;
constexpr uint32_t kModeRing = 1;
constexpr uint32_t kPiece = 4;
// REDUCE (reader): an all-reduce of two islands' partials, a4 pages kept by each
// island: this island keeps pages a8 .., the other a9 ... Ranks 0 .. a5 - 1, share r of
// the kept pages each: its partial (tensor a0, this island's) of the other's kept
// pages gathered into the other's staging tensor a1 (keeper: the same rank there, port
// kPortKept); then, once its own kept share's partials are there (km::gathered), each
// page of its partial and of the staging tensor a2 (this island's) added word by word
// into the result a3 (a copy), and its result share gathered into the other's copy.
// a6: the report (as GATHER's), a7: the compute's busy loop.
constexpr uint32_t kHandlerReduce = 3;
constexpr uint32_t kPortKept = 0;
constexpr uint32_t kFold = 8;  // pages a fold step
// STAMPS (reader, one rank): the lead page CPU's KM_PUT_STAMPS words of link core 0
// toward island a0, and this core's wall clock, into page 0 of tensor a1.
constexpr uint32_t kHandlerStamps = 4;
// FILL (reader): the command's ranks write pages a1 .. a1 + a2 - 1 of tensor a0 (this
// island's pages), word w of page p the pattern p * kPagePrime + w * kWordPrime + a3.
// CHECK (reader): the same pages read and compared with the pattern of salt a3 (a4 not
// 0: plus the pattern of salt a4, a REDUCE's sum); the rank's bad words and first bad
// page into its page of tensor a5 (kCheck*).
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
