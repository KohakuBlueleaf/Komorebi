// Graph example: the handlers and buffers its kernels share. The state is one uint32
// per core, word 0 of a 64-byte page: x[p] is page p's word, and page p is core p's.
#pragma once

#include <stdint.h>

// STEP: a0 state_address, a1 round: x[p] = x[p] + round + p.
constexpr uint32_t kHandlerStep = 1;
// LEFT: a0 state_address: x[p] = x[p] >> 1.
constexpr uint32_t kHandlerLeft = 2;
// RIGHT: a0 state_address: x[p] = x[p] * 3.
constexpr uint32_t kHandlerRight = 3;
// NORM: a0 state_address, a1 pages: the leader's result word is the sum of x[0..pages).
constexpr uint32_t kHandlerNorm = 4;
// STALL (a test): rank a0 stalled_rank waits for the stop flag and does not report.
constexpr uint32_t kHandlerStall = 5;

constexpr uint32_t kCbScratch = 16;
constexpr uint32_t kAskPort = 0;  // the state pages another island holds count here
constexpr uint32_t kPageBytes = 64;
