// The epilogue program's layout: its buffers and the info page's words. Must match
// epilogue.py.
#pragma once

#include <stdint.h>

namespace layout {

    constexpr uint32_t kCbA = 0, kCbB = 1, kCbOut = 16, kCbInfo = 24;

    // Info page words: kNonce is written last; kStream the math CPU's span from its
    // first wait to its last body; kSync + i an empty span; kBody + t tile t's body.
    constexpr uint32_t kNonce = 0, kError = 1, kTiles = 2, kMathDone = 3, kStream = 4;
    constexpr uint32_t kSync = 16, kSyncs = 16, kBody = 256, kBodies = 512;

    constexpr uint32_t kErrorSpin = 1;           // the pack CPU's wait for math gave up
    constexpr uint32_t kSpinLimit = 1u << 24;    // about a second of polls
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

}  // namespace layout
