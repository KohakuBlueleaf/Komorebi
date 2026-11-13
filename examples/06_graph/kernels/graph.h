// Graph example: the handlers and buffers its kernels share. The state is one uint32
// per core, word 0 of a 64-byte page; page p belongs to core p.
#pragma once

#include <stdint.h>

// STEP: a0 state, a1 round number: x[p] = x[p] + a1 + p.
constexpr uint32_t kHandlerStep = 1;
// LEFT: a0 state: x[p] = x[p] >> 1.
constexpr uint32_t kHandlerLeft = 2;
// RIGHT: a0 state: x[p] = x[p] * 3.
constexpr uint32_t kHandlerRight = 3;
// NORM: a0 state, a1 pages: the leader's result word is the sum of x[0..a1).
constexpr uint32_t kHandlerNorm = 4;
// STALL (a test): member a0 waits for the stop flag and does not report.
constexpr uint32_t kHandlerStall = 5;

constexpr uint32_t kCbScratch = 16;
constexpr uint32_t kPageBytes = 64;
