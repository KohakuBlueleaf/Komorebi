// Map/collect example: the segments, buffers and record arguments its kernels share.
#pragma once

#include <stdint.h>

// MAP: a0 input, a1 input tiles T, a2 output, a3 output page of member 0; member a5
// holds its output tile a4 cycles in compute (a test of slow compute).
constexpr uint32_t kSegMap = 1;
// STALL: member a0 waits for the stop word and does not report; the others report.
constexpr uint32_t kSegStall = 2;
// ALIGN: every member reports at wall-clock time a9 (low word; the test feeder sets
// it).
constexpr uint32_t kSegAlign = 3;
// LATE: member a0 reports a1 cycles after it took the record (wall clock).
constexpr uint32_t kSegLate = 4;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
