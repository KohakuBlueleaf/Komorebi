// Map/collect example: the handlers, buffers and command arguments its kernels share.
#pragma once

#include <stdint.h>

// MAP: a0 input, a1 input tiles T from tile a6, a2 output, a3 output page of rank 0;
// rank a5 holds its output tile a4 cycles in compute (a test of slow compute).
constexpr uint32_t kHandlerMap = 1;
// STALL: rank a0 waits for the stop flag and does not report; the others report.
constexpr uint32_t kHandlerStall = 2;
// ALIGN: every core of the group reports at wall-clock time a9 (low word; the test
// feeder sets it).
constexpr uint32_t kHandlerAlign = 3;
// LATE: rank a0 reports a1 cycles after it took the command (wall clock).
constexpr uint32_t kHandlerLate = 4;
// BAD_ITEM (a test): rank a0 asks for an item past the buffer end; a1 = 0: writer,
// 2 pages of kCbOut (after an odd number of pages); 1: reader, kInPages of kCbIn.
constexpr uint32_t kHandlerBadItem = 5;
// RESULT (a test): every core's writer gives a0 + its index as the result; with
// KM_F_RESULT the command is acknowledged with the leader's, a0.
constexpr uint32_t kHandlerResult = 6;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
