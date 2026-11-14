// Map/collect example: the handlers, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// MAP: a0 input_address, a1 tiles (T), a2 output_address, a3 first_page (of rank 0), a6
// first_tile; rank a5 slow_rank holds its output tile a4 slow_cycles cycles (a test).
constexpr uint32_t kHandlerMap = 1;
// MAP's a7: the placement of its input among the reader's two (compile arguments after
// the input's accessor's): kInputFirst, kInputSecond. Pages another island holds are
// asked for (km::read_pages) and counted at port kAskPort.
constexpr uint32_t kInputFirst = 0;
constexpr uint32_t kInputSecond = 1;
constexpr uint32_t kAskPort = 1;
// STALL: rank a0 stalled_rank waits for the stop flag and does not report; the others
// report.
constexpr uint32_t kHandlerStall = 2;
// ALIGN: every core of the group reports at wall-clock time a9 report_time (low word;
// the test feeder sets it).
constexpr uint32_t kHandlerAlign = 3;
// LATE: rank a0 late_rank reports a1 delay cycles after it took the command (wall
// clock).
constexpr uint32_t kHandlerLate = 4;
// BAD_ITEM (a test): rank a0 faulty_rank asks for an item past the buffer end; a1
// faulty_role 0: writer, 2 pages of kCbOut (after an odd number); 1: reader, kInPages.
constexpr uint32_t kHandlerBadItem = 5;
// RESULT (a test): every core's writer gives a0 result_base + its index as the result;
// with KM_FLAG_RESULT the command is acknowledged with the leader's, a0.
constexpr uint32_t kHandlerResult = 6;
// DOT: the block is words w; the writer gives sum w[i] * (a0 x0 + i) mod 2^32 as the
// result, the reader gives up (bad_item) unless the words sum to a1 word_sum.
constexpr uint32_t kHandlerDot = 7;
// HANG (a test): rank a0 hung_rank waits, by a1 hang_mode, for what never comes; the
// dispatcher gives it up at its deadline and the core takes the next command.
constexpr uint32_t kHandlerHang = 8;
constexpr uint32_t kHangPort = 1;    // the reader, for an arrival at port 0
constexpr uint32_t kHangInput = 2;   // compute, for 2 input tiles (the reader sends 1)
constexpr uint32_t kHangOutput = 3;  // the writer, for 2 output tiles (compute sends 1)
// COUNT (a test): a0 0, every core of the group adds one to its count of COUNTs (a
// word of the writer, 0 at the load); a0 1, the writer gives its count as the result.
constexpr uint32_t kHandlerCount = 9;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
