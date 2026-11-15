// Shared layout of the SiLU body experiment: buffers, the info page's words, the
// bound of the pack CPU's wait. Must match front.py.
#pragma once

#include <stdint.h>

namespace front {

    constexpr uint32_t kCbIn = 0, kCbOut = 16, kCbInfo = 24;
    constexpr uint32_t kMaxTiles = 32;

    // Info page words: kNonce is written last; kBody + t the vector body of tile t,
    // kLoop + t the whole tile (copy, body, pack); kSync + i an empty timed span.
    constexpr uint32_t kNonce = 0, kError = 1, kTiles = 2, kMathDone = 3;
    constexpr uint32_t kBody = 64, kLoop = 128, kSync = 192, kSyncs = 16;

    constexpr uint32_t kErrorSpin = 1;           // the pack CPU's wait for math gave up
    constexpr uint32_t kSpinLimit = 1u << 24;    // about a second of polls
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

}  // namespace front
