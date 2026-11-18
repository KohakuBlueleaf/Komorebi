// What all five processors run alike: LOOP's integer loop and the two sides of a PING
// round trip through L1, built with one optimisation level (five.py).
#pragma once

#include <stdint.h>

#include "five.h"
#include "komorebi/device/core/km_core.hpp"

namespace five {

    inline volatile tt_l1_ptr uint32_t* words(uint32_t address) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    }

    // `iterations` steps over the 1,024 words of `block`: read a word, mix it into the
    // sum with shifts, adds and xors (no multiply), write it back. Returns the sum.
    __attribute__((noinline)) uint32_t
    mix(volatile tt_l1_ptr uint32_t* block, uint32_t iterations, uint32_t sum) {
        for (uint32_t i = 0; i < iterations; i++) {
            const uint32_t k = i & (kBlockWords - 1);
            sum = ((sum << 5) + sum) ^ block[k] ^ i;
            block[k] = sum;
        }
        return sum;
    }

    // LOOP on processor `p`: a1 iterations, or (a2 not 0) steps of 4,096 until a2 x
    // 1,024 cycles have passed; start, end, sum and iterations into p's result block.
    inline void loop(
        uint32_t rows, uint32_t blocks, uint32_t p, const km_command_t& c
    ) {
        if (((c.a0 >> p) & 1) == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* r = words(rows) + kResult + p * kResultWords;
        volatile tt_l1_ptr uint32_t* block = words(blocks) + p * kBlockWords;
        const uint64_t start = km::wall64();
        uint32_t sum = 0x9E3779B9u, done = 0;
        if (c.a2 == 0) {
            sum = mix(block, c.a1, sum);
            done = c.a1;
        } else {
            const uint64_t until = start + (static_cast<uint64_t>(c.a2) << 10);
            while (km::wall64() < until) {
                sum = mix(block, 4096, sum);
                done += 4096;
            }
        }
        const uint64_t end = km::wall64();
        r[kRStart] = static_cast<uint32_t>(start);
        r[kRStart + 1] = static_cast<uint32_t>(start >> 32);
        r[kREnd] = static_cast<uint32_t>(end);
        r[kREnd + 1] = static_cast<uint32_t>(end >> 32);
        r[kRSum] = sum;
        r[kRIterations] = done;
    }

    // Waits until *w == want, at most `limit` cycles. False: it ran out.
    inline bool wait_word(
        volatile tt_l1_ptr uint32_t* w, uint32_t want, uint32_t limit
    ) {
        const uint32_t t0 = km::wall();
        for (;;) {
            invalidate_l1_cache();
            if (*w == want) {
                return true;
            }
            if (km::wall() - t0 > limit) {
                return false;
            }
        }
    }

    constexpr uint32_t kRoundLimit = 1000000;  // cycles a round trip may take

    // The first of a PING pair: a1 round trips; ticks, round trips and timeouts into
    // processor p's result block.
    inline void ping(uint32_t rows, uint32_t p, const km_command_t& c) {
        volatile tt_l1_ptr uint32_t* row = words(rows);
        volatile tt_l1_ptr uint32_t* r = row + kResult + p * kResultWords;
        const uint32_t tag = c.sequence << 16;
        uint32_t k = 1, missed = 0;
        const uint64_t t0 = km::wall64();
        for (; k <= c.a1; k++) {
            row[kPing] = tag | k;
            if (!wait_word(row + kPong, tag | k, kRoundLimit)) {
                missed++;
                break;
            }
        }
        const uint64_t ticks = km::wall64() - t0;
        r[kRPing] = static_cast<uint32_t>(ticks);
        r[kRPing + 1] = static_cast<uint32_t>(ticks >> 32);
        r[kRPing + 2] = k - 1;
        r[kRPing + 3] = missed;
    }

    // The second of a PING pair: answers each ping k with k.
    inline void pong(uint32_t rows, const km_command_t& c) {
        volatile tt_l1_ptr uint32_t* row = words(rows);
        const uint32_t tag = c.sequence << 16;
        for (uint32_t k = 1; k <= c.a1; k++) {
            if (!wait_word(row + kPing, tag | k, kRoundLimit)) {
                return;
            }
            row[kPong] = tag | k;
        }
    }

}  // namespace five
