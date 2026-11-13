// Shared layout of the tile probes: input and output words, error codes, probe ids.
// Must match probes.py.
#pragma once

#include <stdint.h>

namespace probe {

    // Probe ids (define PROBE).
    constexpr uint32_t kSmoke = 0;
    constexpr uint32_t kPred = 1;
    constexpr uint32_t kAnyCC = 2;
    constexpr uint32_t kGather = 3;
    constexpr uint32_t kSlab = 4;
    constexpr uint32_t kRiscv = 5;
    constexpr uint32_t kWalk = 6;
    constexpr uint32_t kNocRead = 7;

    // Output words. kNonce is written last: the host sees a finished run by it.
    constexpr uint32_t kNonce = 0, kError = 1, kNumTimes = 2;
    constexpr uint32_t kDone0 = 4;   // kDone0 + t: done word of TRISC t (probe 5)
    constexpr uint32_t kTimes = 16;  // timing words (wall-clock cycles)
    constexpr uint32_t kMaxTimes = 2032;
    constexpr uint32_t kData = 2048;  // probe data
    constexpr uint32_t kOutWords = 8192;

    // Input words: probe data from word 0. kInWords is the CB page.
    constexpr uint32_t kInWords = 8192;

    // Error codes (bit set per failure; 0 is a clean run).
    constexpr uint32_t kErrorSpin = 1;  // a timed wait gave up
    constexpr uint32_t kErrorLoop = 2;  // a data-dependent loop hit its bound

    // CBs: the input page, the output page, scratch L1 for the RISC-V loops.
    constexpr uint32_t kCbIn = 0, kCbOut = 16, kCbScratch = 24;
    constexpr uint32_t kScratchWords = 4096;

    // Timeout of every wait of this kernel: about 2^24 x 20 cycles, under a second.
    constexpr uint32_t kSpinLimit = 1u << 24;

}  // namespace probe
