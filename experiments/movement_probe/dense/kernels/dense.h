// The matrix product's handlers, command words, buffers and profile page (a copy of
// the program's, plus Y's second output half). Sizes are in 32 x 32 tiles.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_broadcast.hpp"

// LOAD: a0 W (DRAM, [K, N] tiles row-major), a1 the W buffer, a3 kt, a4 nt. Every
// core reads all of W into that buffer and keeps it for the commands that follow.
constexpr uint32_t kHandlerLoad = 1;
// PRODUCT: a0 A ([M, K]), a1 Y ([M, N]), a2 mt, a3 kt, a4 nt, a5 kc (K chunk), a6 ct
// (column block), a7 mode, a8 form, a9 W (form kFormStream). Rank i: rows
// partition(mt).
constexpr uint32_t kHandlerProduct = 2;
// PROF: a0 profile tensor; the writer stores this core's profile page at page `me`.
constexpr uint32_t kHandlerProf = 3;
// LAYER: a0 A, a1 Y, a2 mt, a3 kt (one chunk), a4 nt, a5 ct, a6 W buffer (0: the gate
// pass), a7 the epilogue, a8 A | B << 8 | Y << 16: a held buffer, or kDram for A, Y.
constexpr uint32_t kHandlerLayer = 4;
// Epilogues: none; silu; times B; the gate pass silu(A) * B (no product, kt == nt).
constexpr uint32_t kEpNone = 0;
constexpr uint32_t kEpSilu = 1;
constexpr uint32_t kEpMul = 2;
constexpr uint32_t kEpGate = 3;
constexpr uint32_t kDram = 0xFF;

// PRODUCT modes: the full product, or one half of it with the other left out.
constexpr uint32_t kModeFull = 0;
constexpr uint32_t kModeReadOnly = 1;  // A read and popped; no product, no writes
// A from L1 as it is, no reads; packed, not written.
constexpr uint32_t kModeComputeOnly = 2;
// Forms: W resident from LOAD, or W streamed per column block in K chunks.
constexpr uint32_t kFormResident = 0;
constexpr uint32_t kFormStream = 1;

// Buffers: A items (a K chunk, or the whole row when it is one chunk), resident W,
// streamed W chunks, output column blocks, the profile page.
constexpr uint32_t kCbA = 0;
constexpr uint32_t kCbW = 1;
constexpr uint32_t kCbWs = 2;
constexpr uint32_t kCbOut = 16;
// Y's paired output buffer: kCbOut is mover B's half (the writer role's CPU), this
// mover A's; Y_PATTERN (a pattern word, default "B") names the owner of each block.
constexpr uint32_t kOutputMoverA = 10;
#if !defined(Y_PATTERN)
#define Y_PATTERN 0x101
#endif
constexpr uint32_t kCbProf = 3;
constexpr uint32_t kCbLoaded = 4;  // reader to writer: W has landed (LOAD completes)
// LAYER: held intermediates (a core's rows, packed by one command, taken by the
// next), the second and third resident W.
constexpr uint32_t kCbHold0 = 5;
constexpr uint32_t kCbHold1 = 6;
constexpr uint32_t kCbHold2 = 7;
constexpr uint32_t kCbW3 = 8;
constexpr uint32_t kCbW2 = 9;
constexpr uint32_t kLoadedBytes = 64;
constexpr uint32_t kTileBytes = 2048;  // bf16
constexpr uint32_t kProfBytes = 1024;

// Profile page (32-bit words; cycles of the wall clock, summed over commands). Reader:
// waits for A space, reads (submit and barrier), whole bodies, tiles read.
constexpr uint32_t kPrRdWait = 0;
constexpr uint32_t kPrRdRead = 1;
constexpr uint32_t kPrRdBody = 2;
constexpr uint32_t kPrRdTiles = 3;
constexpr uint32_t kPrRdWsWait = 4;
constexpr uint32_t kPrRdWsRead = 5;
// Compute: unpack waits for A, unpack waits for W chunks, unpack bodies; math in the
// product (acquire to commit); pack waits for math, for output space, packing, bodies.
constexpr uint32_t kPrUnWaitA = 8;
constexpr uint32_t kPrUnWaitW = 9;
constexpr uint32_t kPrUnBody = 10;
constexpr uint32_t kPrMathMm = 12;
constexpr uint32_t kPrMathBody = 13;
constexpr uint32_t kPrMathEp = 14;  // LAYER: the epilogue (silu, times B, the gate)
constexpr uint32_t kPrPkWaitMath = 16;
constexpr uint32_t kPrPkWaitOut = 17;
constexpr uint32_t kPrPkPack = 18;
constexpr uint32_t kPrPkBody = 19;
constexpr uint32_t kPrPkTiles = 20;
// Writer: waits for output, writes (submit and flush), bodies, tiles written.
constexpr uint32_t kPrWrWait = 24;
constexpr uint32_t kPrWrWrite = 25;
constexpr uint32_t kPrWrBody = 26;
constexpr uint32_t kPrWrTiles = 27;
constexpr uint32_t kProfWords = 32;

#if !defined(COMPILE_FOR_TRISC)

// Timed wait until this kernel's reads have landed (kFlushReads) or its writes have
// left L1 (kFlushSent); a give-up goes into state word `word`.
inline bool flushed(const km::Core& c, uint32_t word, uint32_t what) {
    const uint32_t code = c.poll(
        [&] {
            if constexpr (noc_mode == DM_DYNAMIC_NOC) {
                return (
                    what == km::kFlushReads
                        ? ncrisc_dynamic_noc_reads_flushed(noc_index)
                        : ncrisc_dynamic_noc_nonposted_writes_sent(noc_index)
                );
            } else {
                return (
                    what == km::kFlushReads
                        ? ncrisc_noc_reads_flushed(noc_index)
                        : ncrisc_noc_nonposted_writes_sent(noc_index)
                );
            }
        },
        c.args().wait_limit,
        c.args().wait_limit
    );
    if (code != km::kGaveNone) {
        c.gave_up(word, code, km::kWhereData, 0, 0);
    }
    return code == km::kGaveNone;
}

#endif
