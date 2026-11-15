// Shared layout of the tile-rules experiment: buffers, the info page's words, the
// bound of the pack CPU's wait. Must match rules.py.
#pragma once

#include <stdint.h>

namespace rules {

    constexpr uint32_t kCbA = 0, kCbB = 1, kCbP = 2, kCbOut = 16, kCbInfo = 24;
    // The hand norm's buffers: its one-page buffers, the row again for the products
    // (decode's DC_CB_Y: not unpacked to Dst); its P pages (decode's constant tiles).
    constexpr uint32_t kCbSquares = 3, kCbScale = 4, kCbY = 5;
    constexpr uint32_t kPIdentity = 0, kPMean = 1, kPWeight = 2;
    // The per-core attention body's buffers: a step's K and V tiles, the query row,
    // P, the row state, the ones tile (the hand body's row sum on the matrix unit).
    constexpr uint32_t kCbKv = 0, kCbQ = 1, kCbProb = 2, kCbSt = 3, kCbOnes = 4;
    // Its info words: the step's score products, vector body and P V products.
    constexpr uint32_t kScore = 64, kVector = 256, kProduct = 320;
    constexpr uint32_t kMaxUnits = 32;

    // Info page words: kNonce is written last; kBody + u the body of unit u (a tile,
    // or a row of tiles), kLoop + u the whole unit; kSync + i an empty timed span.
    constexpr uint32_t kNonce = 0, kError = 1, kUnits = 2, kMathDone = 3;
    constexpr uint32_t kBody = 64, kLoop = 128, kSync = 192, kSyncs = 16;

    constexpr uint32_t kErrorSpin = 1;           // the pack CPU's wait for math gave up
    constexpr uint32_t kSpinLimit = 1u << 24;    // about a second of polls
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

}  // namespace rules
