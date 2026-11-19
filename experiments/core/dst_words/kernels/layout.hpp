// Shared layout of the Dst words probe: buffers and the info page's words. Must match
// probe.py.
#pragma once

#include <stdint.h>

namespace dst_words {

    constexpr uint32_t kCbIn = 0, kCbOut = 16, kCbInfo = 24;
    // Info words: kNonce, then per input tile t its write (into Dst) and read (out of
    // Dst) spans in wall-clock ticks at kWrite + t and kRead + t.
    constexpr uint32_t kNonce = 0, kTiles = 1, kWrite = 16, kRead = 64;
    constexpr uint32_t kMaxTiles = 48;
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

}  // namespace dst_words
