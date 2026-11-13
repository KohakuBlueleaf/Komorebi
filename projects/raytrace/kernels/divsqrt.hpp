// Division and square root rounded to nearest on the vector unit, bit for bit as
// float32 on the host: an estimate within an ulp, then an exact remainder test.
#pragma once

#include <stdint.h>

#include "sfpu.hpp"

namespace rt {

    constexpr uint32_t kMagicRsqrt = 0x5F3759DF;  // 1 / sqrt(x) estimate from the bits
    constexpr uint32_t kHalfB = 0x3F00, kThreeHalvesB = 0x3FC0;  // 0.5, 1.5 as bfloat16
    constexpr uint32_t kFloatAbs = 1, kNoBias = 0, kPad9 = 1;

    // m = 2^g Q B mod 2^32 from mantissas q, b (hidden bits set), from the 23-bit
    // product halves (as recip() does). Uses t; q and b kept.
    inline void prod_scaled(
        uint32_t q, uint32_t b, uint32_t m, uint32_t t, uint32_t g
    ) {
        TT_SFPMUL24(q, b, ZERO, m, 1);  // high bits of the 23-bit product
        TT_SFPMUL24(q, b, ZERO, t, 0);  // its low 23 bits
        TTI_SFPNOP;
        TT_SFPIADD(0, q, m, kCcNone);
        TT_SFPIADD(0, b, m, kCcNone);
        TT_SFPSHFT(23 + g, 0, m, 1);
        TT_SFPSHFT(g, 0, t, 1);
        TT_SFPIADD(0, t, m, kCcNone);
    }

    // r one ulp up where rg (2^g times the exact remainder) exceeds `half`; `half`
    // becomes the lower threshold, halved below a power of two. Uses lo.
    inline void round_fix(uint32_t r, uint32_t rg, uint32_t half, uint32_t lo) {
        TT_SFPMOV(0, rg, lo, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, half, lo, kTwos);  // half - rg < 0: one up
        TT_SFPIADD(1, r, r, kImmediate | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
        // The lower neighbour of a power of two is half an ulp away.
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPEXMAN(0, r, lo, kPad9);
        TT_SFPSETCC(0, lo, 0, kEq0);
        TT_SFPSHFT(0xFFF, 0, half, 1);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // q = a / b rounded to nearest: a in L0 (finite; +-0 gives +-0), b in L1 (positive,
    // in recip's range), the quotient normal or zero. q in L2; L0, L1 kept; L3 to L7.
    inline void divide() {
        recip(L1, L3, L4, L5, L6, L7);  // y = 1 / b
        TTI_SFPABS(0, L0, L4, kFloatAbs);
        mul(L4, L3, L2);                   // q = |a| y
        TT_SFPMAD(L2, L1, L4, L5, kNegA);  // |a| - q b, the product kept wider
        TT_SFPMAD(L5, L3, L2, L2, 0);      // q += (|a| - q b) y
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPSETCC(0, L4, 0, kNe0);  // a = +-0: q is +0 already
        // 64 R = 64 (A 2^k - Q B) mod 2^32, k + 6 = ea - eb - eq + 29
        TT_SFPEXEXP(0, L4, L3, kNoBias);
        TT_SFPEXEXP(0, L1, L5, kNoBias);
        TT_SFPIADD(0, L3, L5, kTwos | kCcNone);  // ea - eb
        TT_SFPEXEXP(0, L2, L3, kNoBias);
        TT_SFPIADD(0, L5, L3, kTwos | kCcNone);  // - eq
        TT_SFPIADD(29, L3, L3, kImmediate | kCcNone);
        TT_SFPEXMAN(0, L4, L5, 0);
        TT_SFPSHFT(0, L3, L5, 0);  // A << (k + 6)
        TT_SFPEXMAN(0, L2, L3, 0);
        TT_SFPEXMAN(0, L1, L4, 0);
        prod_scaled(L3, L4, L6, L7, 6);
        TT_SFPIADD(0, L5, L6, kTwos | kCcNone);  // 64 R
        TT_SFPSHFT(5, 0, L4, 1);                 // 32 B
        round_fix(L2, L6, L4, L7);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, L6, L4, 0);  // 64 R + below < 0: one down
        TT_SFPIADD(0xFFF, L2, L2, kImmediate | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
        TT_SFPMOV(0, L0, L3, 0);
        TT_SFPSETSGN(0, L2, L3, 0);  // |q| with a's sign
        TT_SFPMOV(0, L3, L2, 0);
    }

    // s = sqrt(x) rounded to nearest: x in L0 (positive, exponent at least -100), s in
    // L1; L0 kept; L2 to L7.
    inline void square_root() {
        TT_SFPMOV(0, L0, L2, 0);
        TT_SFPSHFT(0xFFF, 0, L2, 1);  // x >> 1
        immediate(L3, kMagicRsqrt);
        TT_SFPIADD(0, L3, L2, kTwos | kCcNone);  // y = magic - (x >> 1)
        TTI_SFPLOADI(L4, kFloatB, kHalfB);
        TTI_SFPLOADI(L5, kFloatB, kThreeHalvesB);
        mul(L0, L4, L3);  // x / 2
#pragma GCC unroll 3
        for (uint32_t k = 0; k < 3; ++k) {
            mul(L2, L2, L6);
            TT_SFPMAD(L3, L6, L5, L6, kNegA);  // 1.5 - (x / 2) y y
            mul(L2, L6, L2);
        }
        mul(L0, L2, L1);                   // s = x y
        TT_SFPMAD(L1, L1, L0, L6, kNegA);  // x - s s
        mul(L2, L4, L7);                   // y / 2
        TT_SFPMAD(L6, L7, L1, L1, 0);      // s += (x - s s) y / 2
        // 32 R = 32 (X 2^k - S S) mod 2^32, k + 5 = ex - 2 es + 28: R is up to 2 S
        // ulps, so 64 R would wrap for S near 2^24.
        TT_SFPEXEXP(0, L0, L2, kNoBias);
        TT_SFPEXEXP(0, L1, L3, kNoBias);
        TT_SFPIADD(0, L2, L3, kTwos | kCcNone);
        TT_SFPEXEXP(0, L1, L2, kNoBias);
        TT_SFPIADD(0, L3, L2, kTwos | kCcNone);
        TT_SFPIADD(28, L2, L2, kImmediate | kCcNone);
        TT_SFPEXMAN(0, L0, L3, 0);
        TT_SFPSHFT(0, L2, L3, 0);  // X << (k + 6)
        TT_SFPEXMAN(0, L1, L4, 0);
        TT_SFPMOV(0, L4, L2, 0);
        prod_scaled(L4, L2, L5, L6, 5);
        TT_SFPIADD(0, L3, L5, kTwos | kCcNone);  // 32 R
        TT_SFPSHFT(5, 0, L4, 1);                 // 32 S: R > S is past the midpoint
        round_fix(L1, L5, L4, L6);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, L5, L4, kCcNone);  // 32 R + below - 1 < 0: one down
        TT_SFPIADD(0xFFF, L4, L4, kImmediate);
        TT_SFPIADD(0xFFF, L1, L1, kImmediate | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

}  // namespace rt
