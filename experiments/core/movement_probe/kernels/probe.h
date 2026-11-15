// The movement probe's handlers, command words, buffers and profile words, shared by
// the host and the three kernels. A page is 2048 bytes (one tile).
#pragma once

#include <stdint.h>

// RUN: a0 input tensor, a1 output tensor, a2 tiles per core and repetition, a3 tiles
// per block, a4 pattern word, a5 repetitions, a6 program, a7 layout, a8 check salt.
constexpr uint32_t kHandlerRun = 1;
// PROFILE: a0 the profile tensor; the writer stores this core's page at page `me`.
constexpr uint32_t kHandlerProfile = 2;

// Programs: DRAM to L1 (compute drops the tiles), L1 to DRAM (compute hands out
// buffer pages), DRAM to DRAM through L1 (no compute).
constexpr uint32_t kProgramRead = 0;
constexpr uint32_t kProgramWrite = 1;
constexpr uint32_t kProgramCopy = 2;
// Layouts: streams through the movement engine on both movers, or the classic roles
// (the reader reads everything, the writer writes everything, blocking calls).
constexpr uint32_t kLayoutEngine = 0;
constexpr uint32_t kLayoutClassic = 1;

// The paired buffer between the movers and compute: mover A's and mover B's half. The
// profile page. Copies: each mover's own half, and the classic reader-to-writer buffer.
constexpr uint32_t kBufferA = 0;
constexpr uint32_t kBufferB = 1;
constexpr uint32_t kBufferProfile = 2;
constexpr uint32_t kBufferCopyA = 3;
constexpr uint32_t kBufferCopyB = 4;
constexpr uint32_t kBufferClassicCopy = 5;
constexpr uint32_t kTileBytes = 2048;
constexpr uint32_t kProfileBytes = 1024;

// Profile words (32-bit; wall-clock cycles of the last RUN, overwritten by each RUN).
// Per mover: start, end, tiles moved, cycles inside NoC calls and barriers.
constexpr uint32_t kProfileMoverA = 0;
constexpr uint32_t kProfileMoverB = 4;
constexpr uint32_t kProfileStart = 0;
constexpr uint32_t kProfileEnd = 1;
constexpr uint32_t kProfileTiles = 2;
constexpr uint32_t kProfileBusy = 3;
// Compute: start, end, the sum of every word taken, the sum weighted by tile position.
constexpr uint32_t kProfileComputeStart = 8;
constexpr uint32_t kProfileComputeEnd = 9;
constexpr uint32_t kProfileSum = 10;
constexpr uint32_t kProfileWeighted = 11;
constexpr uint32_t kProfileComputeTiles = 12;
constexpr uint32_t kProfileWords = 16;
