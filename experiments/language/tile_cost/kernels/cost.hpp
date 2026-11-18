// Shared layout of the tile-cost probe: buffers, the info page's words, the wall clock,
// the tile's performance counters. Must match probe.py.
#pragma once

#include <stdint.h>

namespace cost {

    constexpr uint32_t kCbA = 0, kCbB = 1, kCbAlt = 2;
    constexpr uint32_t kCbOut = 16, kCbOutAlt = 17, kCbInfo = 24;

    // Info page words; repeat r owns kRep + r * kRepStride: six stamps, the counter
    // read time, the four banks' reference cycles, then the banks' counters.
    constexpr uint32_t kNonce = 0, kError = 1, kRepeats = 2, kTiles = 3, kThrottle = 4;
    constexpr uint32_t kDbgDisable = 5, kBar = 8, kFail = 12, kSync = 16, kSyncs = 16;
    constexpr uint32_t kRep = 64, kRepStride = 112, kMaxRepeats = 8;
    constexpr uint32_t kStamps = 0, kReadTime = 6, kRefs = 8, kCounters = 12;

    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L
    constexpr uint32_t kSpinLimit = 1u << 26;

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    // A bank's debug registers (tt-llk perf_counters/registers.h): reference period,
    // mode and select, control, read-out.
    struct Bank {
        uint32_t ref_period, mode, control, out_l, out_h;
    };

    constexpr uint32_t kDebug = 0xFFB12000;
    constexpr Bank kBanks[4] = {
        {kDebug + 0x0,
         kDebug + 0x4,
         kDebug + 0x8,
         kDebug + 0x100,
         kDebug + 0x104},  // INSTRN_THREAD
        {
            kDebug + 0x18, kDebug + 0x1C, kDebug + 0x20, kDebug + 0x120, kDebug + 0x124
        },  // FPU
        {
            kDebug + 0xC, kDebug + 0x10, kDebug + 0x14, kDebug + 0x108, kDebug + 0x10C
        },  // TDMA_UNPACK
        {
            kDebug + 0xF0, kDebug + 0xF4, kDebug + 0xF8, kDebug + 0x110, kDebug + 0x114
        },  // TDMA_PACK
    };
    constexpr uint32_t kDbgFeatureDisable = kDebug + 0x68;
    constexpr uint32_t kArcScratch = 16;  // MEM_L1_ARC_FW_SCRATCH: the matmul throttle word

    inline void put(uint32_t address, uint32_t value) {
        *reinterpret_cast<volatile uint32_t*>(address) = value;
    }

    inline uint32_t get(uint32_t address) {
        return *reinterpret_cast<volatile uint32_t*>(address);
    }

    // tt-llk hw.h configure + start: the rising edge of control starts, and zeroes, the
    // counts.
    inline void start_bank(const Bank& bank) {
        put(bank.ref_period, 0xFFFFFFFFu);
        put(bank.mode, 0);
        put(bank.control, 0);
        put(bank.control, 1);
    }

    inline void stop_bank(const Bank& bank) {
        put(bank.control, 0);
        put(bank.control, 2);
    }

}  // namespace cost
