// Map/collect example: the handlers, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// MAP: a0 input_address, a1 tiles (T), a2 output_address, a3 first_page (of rank 0), a6
// first_tile; rank a5 slow_rank holds its output tile a4 slow_cycles cycles (a test).
constexpr uint32_t kHandlerMap = 1;
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

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
