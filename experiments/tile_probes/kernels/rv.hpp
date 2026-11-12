// Probe 5: speed of a compute RISC-V core (integer, L1 load / store, software float);
// TRISC t runs it if run-mask bit t is set: results out[kData + 256 t], done kDone0+t.
#pragma once

#include <stdint.h>

#include "probe.hpp"

namespace rv {

    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L

    // Result words per thread (offset in its 256-word block).
    constexpr uint32_t kStart = 0, kEnd = 1;  // wall clock around all loops
    constexpr uint32_t kCycles = 8;           // + loop: cycles of loop i
    constexpr uint32_t kCheck = 32;           // + loop: checksum of loop i
    constexpr uint32_t kLoops = 6;

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    // a += i ^ b; b += a  (add / xor chain)
    [[gnu::noinline]] uint32_t loop_add(uint32_t n) {
        uint32_t a = 1, b = 2;
        for (uint32_t i = 0; i < n; ++i) {
            a += i ^ b;
            b += a;
            asm volatile("" : "+r"(a), "+r"(b));
        }
        return a ^ b;
    }

    // a = a * 0x9E3779B1 + i  (multiply chain)
    [[gnu::noinline]] uint32_t loop_mul(uint32_t n) {
        uint32_t a = 1;
        for (uint32_t i = 0; i < n; ++i) {
            a = a * 0x9E3779B1u + i;
            asm volatile("" : "+r"(a));
        }
        return a;
    }

    // s += p[i % m]  (L1 loads, sequential; four words share a 16-byte L0 line)
    [[gnu::noinline]] uint32_t loop_load(
        const volatile uint32_t* p, uint32_t n, uint32_t m
    ) {
        uint32_t s = 0;
        for (uint32_t i = 0; i < n; ++i) {
            s += p[i & (m - 1)];
        }
        return s;
    }

    // p[i % m] = i  (L1 stores, sequential)
    [[gnu::noinline]] uint32_t loop_store(
        volatile uint32_t* p, uint32_t n, uint32_t m
    ) {
        for (uint32_t i = 0; i < n; ++i) {
            p[i & (m - 1)] = i;
        }
        asm volatile("fence" ::: "memory");
        return p[(n - 1) & (m - 1)];
    }

    // q[j] = p[j] + 1 over m words, n / m times (L1 load, add, store)
    [[gnu::noinline]] uint32_t loop_copy(
        const volatile uint32_t* p, volatile uint32_t* q, uint32_t n, uint32_t m
    ) {
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t j = i & (m - 1);
            q[j] = p[j] + 1;
        }
        asm volatile("fence" ::: "memory");
        uint32_t s = 0;
        for (uint32_t j = 0; j < m; ++j) {
            s += q[j];
        }
        return s;
    }

    // x = x * a + b in software float (kernels are built without hardware float)
    [[gnu::noinline]] uint32_t loop_float(
        uint32_t n, uint32_t a_bits, uint32_t b_bits
    ) {
        float x = 1.0f, a, b;
        __builtin_memcpy(&a, &a_bits, 4);
        __builtin_memcpy(&b, &b_bits, 4);
        for (uint32_t i = 0; i < n; ++i) {
            x = x * a;
            x = x + b;
            asm volatile("" : "+r"(x));
        }
        uint32_t r;
        __builtin_memcpy(&r, &x, 4);
        return r;
    }

    // Runs the loops selected by `loops` (bit i: loop i) with n integer iterations and
    // n_float float iterations; scratch: 4096 L1 words (two halves for the copy).
    inline void run(
        uint32_t t,
        volatile uint32_t* out,
        volatile uint32_t* scratch,
        uint32_t loops,
        uint32_t n,
        uint32_t n_float,
        uint32_t a_bits,
        uint32_t b_bits
    ) {
        volatile uint32_t* r = out + probe::kData + 256 * t;
        volatile uint32_t* mine = scratch + 1024 * t;  // 1024 words per thread
        for (uint32_t j = 0; j < 1024; ++j) {
            mine[j] = j * 3 + t;
        }
        asm volatile("fence" ::: "memory");
        r[kStart] = wall();
        for (uint32_t i = 0; i < kLoops; ++i) {
            if (!((loops >> i) & 1)) {
                continue;
            }
            const uint32_t w0 = wall();
            uint32_t c = 0;
            switch (i) {
                case 0:
                    c = loop_add(n);
                    break;
                case 1:
                    c = loop_mul(n);
                    break;
                case 2:
                    c = loop_load(mine, n, 1024);
                    break;
                case 3:
                    c = loop_store(mine, n, 1024);
                    break;
                case 4:
                    c = loop_copy(mine, mine + 512, n, 512);
                    break;
                default: c = loop_float(n_float, a_bits, b_bits);
                    break;
            }
            const uint32_t w1 = wall();
            r[kCycles + i] = w1 - w0;
            r[kCheck + i] = c;
        }
        r[kEnd] = wall();
    }

}  // namespace rv
