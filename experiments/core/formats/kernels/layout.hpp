// Shared layout of the formats probe: buffers, the info page's words, the bound of
// the pack CPU's wait. Must match probe.py.
#pragma once

#include <stdint.h>

namespace formats {

    constexpr uint32_t kCbA = 0, kCbB = 1, kCbOut = 16, kCbInfo = 24;
    // Info words: kNonce last; kProduct + t tile t's product (or copy) on the math
    // CPU, kTranspose + t its Dst transpose, kSync + i an empty span (one sync).
    constexpr uint32_t kNonce = 0, kError = 1, kCount = 2, kMathDone = 3;
    constexpr uint32_t kProduct = 16, kTranspose = 32, kSync = 48, kSyncs = 8;
    constexpr uint32_t kMaxTiles = 8;

    constexpr uint32_t kErrorSpin = 1;           // the pack CPU's wait for math gave up
    constexpr uint32_t kSpinLimit = 1u << 24;    // about a second of polls
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

}  // namespace formats
