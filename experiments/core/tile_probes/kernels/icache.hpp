// Probe 12: a straight loop body of N dependent adds around the size of a compute
// core's instruction cache; TRISC t runs it if run-mask bit t is set.
#pragma once

#include <stdint.h>

#include "probe.hpp"

namespace icache {

    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    // Adds per loop body; a body is 4 N + 8 bytes (the adds, the count, the branch).
    constexpr uint32_t kSizes[] = {64, 120, 124, 126, 128, 136, 192, 256, 480, 520};
    constexpr uint32_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    // x + n N after n passes of the body, the loop start aligned to 64 bytes.
    template <uint32_t N>
    [[gnu::noinline]] uint32_t body(uint32_t n, uint32_t x) {
        asm volatile(
            ".balign 64\n"
            "1:\n"
            ".rept %2\n"
            "addi %0, %0, 1\n"
            ".endr\n"
            "addi %1, %1, -1\n"
            "bnez %1, 1b\n"
            : "+r"(x), "+r"(n)
            : "i"(N)
        );
        return x;
    }

    template <uint32_t I = 0>
    inline uint32_t run_size(uint32_t i, uint32_t n) {
        if constexpr (I < kNumSizes) {
            if (i == I) {
                return body<kSizes[I]>(n, 0);
            }
            return run_size<I + 1>(i, n);
        }
        return 0;
    }

    // Results of TRISC t at out[kData + 256 t]: per size i, cycles at 2 i, the add
    // count at 2 i + 1; one warm-up pass before each timed one.
    inline void run(uint32_t t, volatile uint32_t* out, uint32_t n) {
        volatile uint32_t* r = out + probe::kData + 256 * t;
        for (uint32_t i = 0; i < kNumSizes; ++i) {
            run_size(i, 1);
            const uint32_t w0 = wall();
            const uint32_t x = run_size(i, n);
            const uint32_t w1 = wall();
            r[2 * i] = w1 - w0;
            r[2 * i + 1] = x;
        }
    }

}  // namespace icache
