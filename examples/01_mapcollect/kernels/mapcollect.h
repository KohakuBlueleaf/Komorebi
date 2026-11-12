// Map/collect example: the segments, buffers and record arguments its kernels share.
#pragma once

#include <stdint.h>

// MAP: a0 input address, a1 input tiles T, a2 output address, a3 output page of member
// 0.
constexpr uint32_t kSegMap = 1;
// STALL: member a0 waits for the stop word and does not report; the others report.
constexpr uint32_t kSegStall = 2;

constexpr uint32_t kCbIn = 0;
constexpr uint32_t kCbOut = 16;
constexpr uint32_t kInPages = 16;
constexpr uint32_t kBlock = 8;  // input tiles per read barrier
constexpr uint32_t kTileBytes = 2048;
