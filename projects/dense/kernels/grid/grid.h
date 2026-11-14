// Matrix product over a rectangle of cores: A blocks on the row broadcast, W blocks on
// the column broadcast. Handlers, buffers, arguments, profile follow.
#pragma once

#include <stdint.h>

#include "../dense.h"

// GRID: a0 a_address (A, [M, K] tiles), a1 w_address (W, [K, N]), a2 y_address (Y,
// [M, N]), a3 mode. One call.
constexpr uint32_t kHandlerGrid = 1;
// PROF: a0 profile_address (the profile tensor); the writer stores this core's profile
// page at page `me`.
constexpr uint32_t kHandlerGridProf = 2;

// GRID modes: full; broadcasts only (blocks delivered and taken, no product, no
// writes); product only (no reads or broadcasts, blocks packed from L1, not written).
constexpr uint32_t kGridFull = 0;
constexpr uint32_t kGridBroadcasts = 1;
constexpr uint32_t kGridProduct = 2;

constexpr uint32_t kCbRow = 0;  // row broadcast: B_A blocks of bm x kb tiles of A
constexpr uint32_t kCbCol = 1;  // column broadcast: B_W blocks of kb x bn tiles of W
constexpr uint32_t kCbGridOut = 16;  // output subblocks, bf16: two of sh x sw tiles
constexpr uint32_t kCbPart = 24;     // partial sums of one output block, fp32: bm x bn
constexpr uint32_t kCbGridProf = 3;
constexpr uint32_t kPartBytes = 4096;
constexpr uint32_t kBroadcastRow = 0;  // broadcast ids: rows on NoC0 (reader), columns on NoC1
constexpr uint32_t kBroadcastCol = 1;

// Arguments: reader, writer: broadcast words, then core words; compute: core words.
// Core: included (1 in the rectangle), first row and column tile of its output, plan.
constexpr uint32_t kArgBroadcast = 0;
constexpr uint32_t kArgCore = km::kBroadcastArgs;
constexpr uint32_t kCoreRank = 0;
constexpr uint32_t kCoreM0 = 1;
constexpr uint32_t kCoreN0 = 2;
constexpr uint32_t kCorePlan = 3;
// Plan words: tiles M, K, N; output tiles per core (rows, columns); block bm x bn; K
// block kb, K blocks, tiles of the last; blocks per core (rows, columns); subblock.
constexpr uint32_t kPlMt = 0;
constexpr uint32_t kPlKt = 1;
constexpr uint32_t kPlNt = 2;
constexpr uint32_t kPlPcM = 3;
constexpr uint32_t kPlPcN = 4;
constexpr uint32_t kPlBm = 5;
constexpr uint32_t kPlBn = 6;
constexpr uint32_t kPlKb = 7;
constexpr uint32_t kPlNk = 8;
constexpr uint32_t kPlKbLast = 9;
constexpr uint32_t kPlNbh = 10;
constexpr uint32_t kPlNbw = 11;
constexpr uint32_t kPlSh = 12;
constexpr uint32_t kPlSw = 13;
constexpr uint32_t kPlanWords = 14;
constexpr uint32_t kCoreWords = kCorePlan + kPlanWords;

// Profile page (32-bit words, wall-clock cycles summed over commands). Reader: DRAM
// reads of A blocks (submit), row broadcast calls (reserve and cast, or arrive), body.
constexpr uint32_t kGpRdRead = 0;
constexpr uint32_t kGpRdBroadcast = 1;
constexpr uint32_t kGpRdBody = 2;
// Writer: DRAM reads of W blocks, column broadcast calls, waits for output, output
// writes.
constexpr uint32_t kGpWrRead = 4;
constexpr uint32_t kGpWrBroadcast = 5;
constexpr uint32_t kGpWrWait = 6;
constexpr uint32_t kGpWrWrite = 7;
constexpr uint32_t kGpWrBody = 8;
// Unpack: waits for row blocks, column blocks, partial sums; bodies.
constexpr uint32_t kGpUnWaitA = 12;
constexpr uint32_t kGpUnWaitW = 13;
constexpr uint32_t kGpUnWaitPart = 14;
constexpr uint32_t kGpUnBody = 15;
// Math: the products (acquire to commit); bodies. Pack: waits for room, packing,
// bodies.
constexpr uint32_t kGpMathMm = 16;
constexpr uint32_t kGpMathBody = 17;
constexpr uint32_t kGpPkWaitOut = 20;
constexpr uint32_t kGpPkPack = 21;
constexpr uint32_t kGpPkBody = 22;
constexpr uint32_t kGpWords = 24;
