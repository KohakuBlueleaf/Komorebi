// The frame's stages on the vector unit, 32 pixels at a time: primary rays from the
// camera (GEN), shading and the secondary rays (SHADE1, SHADE2), the colour (IMAGE).
#pragma once

#include <stdint.h>

#include "divsqrt.hpp"
#include "raytrace.h"
#include "sfpu.hpp"

namespace shade {

    using namespace rt;

    // Constants of the reference (ref/raytrace.py) as float32 bits.
    constexpr uint32_t kTmax = 0x7149F2CA;     // 1e30
    constexpr uint32_t kMinDir = 0x2B8CBCCC;   // 1e-12
    constexpr uint32_t kMinG2 = 0x2B8CBCCC;    // 1e-12
    constexpr uint32_t kMinB2 = 0x358637BD;    // 1e-6
    constexpr uint32_t kAmbient = 0x3DCCCCCD;  // 0.1
    constexpr uint32_t kGi = 0x3F000000;       // 0.5
    constexpr uint32_t kHalf = 0x3F000000;
    constexpr uint32_t kOne = 0x3F800000;
    constexpr uint32_t kMinusOne = 0xBF800000;
    constexpr uint32_t kTwo = 0x40000000;
    constexpr uint32_t kUnit24 = 0x33800000;  // 2^-24
    constexpr uint32_t kSign = 0x80000000;
    // The hash (ref/rng.py): multipliers, and the seed of the bounce directions.
    constexpr uint32_t kGolden = 0x9E3779B1, kMul1 = 0x7FEB352D, kMul2 = 0x846CA68B;
    constexpr uint32_t kStream = 0x85EBCA6B, kSeed = 7;

    // Slots: the stage's input pages, output pages; scratch; the lane ramp (dx, dy,
    // pixel offset); the pixel numbers; bounce directions.
    constexpr uint32_t kIn = 0;    // up to 48 vectors of input pages
    constexpr uint32_t kOut = 48;  // up to 32 vectors of output pages
    constexpr uint32_t kW = 80;    // scratch: 30 slots
    constexpr uint32_t kRamp = 110, kPx = 113, kDb = 114;

    // Scratch slot names (kW onwards).
    constexpr uint32_t kP = kW, kN = kW + 3, kO2 = kW + 6, kTl = kW + 9, kG = kW + 12;
    constexpr uint32_t kB = kW + 15, kHitM = kW + 18, kT0 = kW + 19, kT1 = kW + 20;
    constexpr uint32_t kT2 = kW + 21, kT3 = kW + 22, kAlb = kW + 23, kCos = kW + 26;
    constexpr uint32_t kInv = kW + 27;

    // Two-operand steps on slots, one float32 rounding each.
    inline void smul(uint32_t a, uint32_t b, uint32_t d) {
        load(L0, a);
        load(L1, b);
        mul(L0, L1, L2);
        store(L2, d);
    }

    inline void sadd(uint32_t a, uint32_t b, uint32_t d) {
        load(L0, a);
        load(L1, b);
        add(L0, L1, L2);
        store(L2, d);
    }

    inline void ssub(uint32_t a, uint32_t b, uint32_t d) {
        load(L0, a);
        load(L1, b);
        sub(L0, L1, L2);
        store(L2, d);
    }

    // d = a * c, d = c + a, d = c - a, d = a - c with c a constant (bits).
    inline void smul_c(uint32_t a, uint32_t c, uint32_t d) {
        load(L0, a);
        imm(L1, c);
        mul(L0, L1, L2);
        store(L2, d);
    }

    inline void sadd_c(uint32_t a, uint32_t c, uint32_t d) {
        load(L0, a);
        imm(L1, c);
        add(L1, L0, L2);
        store(L2, d);
    }

    inline void ssub_from_c(uint32_t a, uint32_t c, uint32_t d) {
        load(L0, a);
        imm(L1, c);
        sub(L1, L0, L2);
        store(L2, d);
    }

    inline void ssub_c(uint32_t a, uint32_t c, uint32_t d) {
        load(L0, a);
        imm(L1, c);
        sub(L0, L1, L2);
        store(L2, d);
    }

    inline void sdiv(uint32_t a, uint32_t b, uint32_t d) {
        load(L0, a);
        load(L1, b);
        divide();
        store(L2, d);
    }

    inline void ssqrt(uint32_t a, uint32_t d) {
        load(L0, a);
        square_root();
        store(L1, d);
    }

    inline void srecip(uint32_t a, uint32_t d) {
        load(L0, a);
        recip(L0, L1, L2, L3, L4, L5);
        store(L1, d);
    }

    inline void copy(uint32_t a, uint32_t d) {
        load(L0, a);
        store(L0, d);
    }

    inline void fill(uint32_t d, uint32_t bits) {
        imm(L0, bits);
        store(L0, d);
    }

    // d = (a0 b0 + a1 b1) + a2 b2 over three slots each.
    inline void dot(uint32_t a, uint32_t b, uint32_t d) {
        load(L0, a);
        load(L1, b);
        mul(L0, L1, L2);
        load(L0, a + 1);
        load(L1, b + 1);
        mul(L0, L1, L3);
        add(L2, L3, L2);
        load(L0, a + 2);
        load(L1, b + 2);
        mul(L0, L1, L3);
        add(L2, L3, L2);
        store(L2, d);
    }

    // max(a, 0) and min(max(a, 0), 1) into d.
    inline void max0(uint32_t a, uint32_t d) {
        load(L0, a);
        TTI_SFPLOADI(L1, kFloatB, 0);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L0, L1, 1);  // L0 = max
        store(L0, d);
    }

    inline void clamp01(uint32_t a, uint32_t d) {
        max0(a, d);
        load(L0, d);
        imm(L1, kOne);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L1, L0, 1);  // L0 = min
        store(L0, d);
    }

    // Lanes enabled where slot s has its sign bit set (a hit flag, an index of -1);
    // between push and pop.
    inline void where_negative(uint32_t s) {
        load(L0, s);
        TTI_SFPSETCC(0, L0, 0, kLt0);
    }

    // Sky colour of the directions in slots d..d+2 into slots c..c+2 (ref sky()).
    inline void sky(uint32_t d, uint32_t c, const uint32_t* cn) {
        dot(d, d, kT0);
        ssqrt(kT0, kT1);
        sdiv(d + 1, kT1, kT2);
        clamp01(kT2, kT2);
        for (uint32_t k = 0; k < 3; ++k) {
            smul_c(kT2, cn[kCnSky + 3 + k], kT3);
            sadd_c(kT3, cn[kCnSky + k], c + k);
        }
    }

    // 1 / d as the reference clamps it: |d| below 1e-12 becomes 1e-12 with d's sign.
    inline void inverse_dir(uint32_t d, uint32_t inv) {
        for (uint32_t k = 0; k < 3; ++k) {
            load(L0, d + k);
            TTI_SFPABS(0, L0, L1, kFloatAbs);
            imm(L2, kMinDir);
            TTI_SFPPUSHC(0, 0, 0, 0);
            TTI_SFPGT(0, L1, L2, kSetCC);  // 1e-12 > |d|
            TT_SFPMOV(0, L0, L3, 0);
            TT_SFPSETSGN(0, L2, L3, 0);
            TT_SFPMOV(0, L3, L0, 0);
            TTI_SFPPOPC(0, 0, 0, 0);
            recip(L0, L1, L2, L3, L4, L5);
            store(L1, inv + k);
        }
    }

    // Normal of the hit (slots n0..) turned against the ray d into n; albedo of the
    // vertex colours c (9 slots) at (u, v) into alb (ref surface()).
    inline void surface(
        uint32_t n0,
        uint32_t c,
        uint32_t u,
        uint32_t v,
        uint32_t d,
        uint32_t n,
        uint32_t alb
    ) {
        dot(n0, d, kT0);
        for (uint32_t k = 0; k < 3; ++k) {
            copy(n0 + k, n + k);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kT0);
        TTI_SFPGT(0, ZERO, L0, kSetCC);  // dot(n, d) > 0
        for (uint32_t k = 0; k < 3; ++k) {
            smul_c(n0 + k, kMinusOne, n + k);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        ssub_from_c(u, kOne, kT1);  // (1 - u) - v
        ssub(kT1, v, kT1);
        for (uint32_t k = 0; k < 3; ++k) {
            smul(c + k, kT1, kT2);
            smul(c + 3 + k, u, kT3);
            sadd(kT2, kT3, kT2);
            smul(c + 6 + k, v, kT3);
            sadd(kT2, kT3, alb + k);
        }
    }

    // From the hit point p and normal n: the shadow ray's origin o2 = p + n eps, its
    // direction tl = light - o2, and the cosine dot(n, tl) / |tl| (ref direct_light()).
    inline void light_ray(
        uint32_t p,
        uint32_t n,
        uint32_t o2,
        uint32_t tl,
        uint32_t cs,
        const uint32_t* cn
    ) {
        for (uint32_t k = 0; k < 3; ++k) {
            smul_c(n + k, cn[kCnEps], kT0);
            sadd(p + k, kT0, o2 + k);
            ssub_from_c(o2 + k, cn[kCnLight + k], tl + k);
        }
        dot(tl, tl, kT0);
        ssqrt(kT0, kT1);
        srecip(kT1, kT2);
        dot(n, tl, kT0);
        smul(kT0, kT2, cs);
    }

    // p = o + d t.
    inline void hit_point(uint32_t o, uint32_t d, uint32_t t, uint32_t p) {
        for (uint32_t k = 0; k < 3; ++k) {
            smul(d + k, t, kT0);
            sadd(o + k, kT0, p + k);
        }
    }

    // L0 = L0 * c mod 2^32 (c a constant), from 16-bit halves and 23-bit products.
    inline void mul32(uint32_t c) {
        imm(L1, 0xFFFF);
        TTI_SFPAND(0, L0, L1, 0);  // x low
        TT_SFPMOV(0, L0, L2, 0);
        TT_SFPSHFT(0xFF0, 0, L2, 1);  // x high (x >> 16)
        imm(L3, c & 0xFFFF);
        imm(L4, c >> 16);
        TT_SFPMUL24(L1, L3, ZERO, L5, 0);  // low 23 bits of xl cl
        TT_SFPMUL24(L1, L3, ZERO, L6, 1);  // bits 23 on
        TT_SFPMUL24(L1, L4, ZERO, L7, 0);  // xl ch (its low 16 bits count)
        TT_SFPMUL24(L2, L3, ZERO, L3, 0);  // xh cl
        TTI_SFPNOP;
        TTI_SFPNOP;
        TT_SFPIADD(0, L3, L7, kCcNone);
        TT_SFPSHFT(16, 0, L7, 1);
        TT_SFPSHFT(23, 0, L6, 1);
        TT_SFPIADD(0, L6, L5, kCcNone);
        TT_SFPIADD(0, L7, L5, kCcNone);
        TT_SFPMOV(0, L5, L0, 0);
    }

    // L0 ^= L0 >> s.
    inline void xorshift(uint32_t s) {
        TT_SFPMOV(0, L0, L1, 0);
        TT_SFPSHFT((-s) & 0xFFF, 0, L1, 1);
        TTI_SFPXOR(0, L1, L0, 0);
    }

    // Slot u = uniform(pixel, stream, seed) of the pixels in slot px (ref rng.py).
    inline void uniform(uint32_t px, uint32_t stream, uint32_t u) {
        load(L0, px);
        mul32(kGolden);
        imm(L1, (stream + 1) * kStream + kSeed);
        TT_SFPIADD(0, L1, L0, kCcNone);
        xorshift(16);
        mul32(kMul1);
        xorshift(15);
        mul32(kMul2);
        xorshift(16);
        TT_SFPSHFT(0xFF8, 0, L0, 1);  // >> 8: 24 bits
        TT_SFPCAST(L0, L0, 0);
        TTI_SFPNOP;
        imm(L1, kUnit24);
        mul(L0, L1, L2);
        store(L2, u);
    }

    // Unit bounce directions db of the pixels in slot px about the normals n (ref
    // bounce_directions()).
    inline void bounce_dir(uint32_t px, uint32_t n, uint32_t db) {
        for (uint32_t k = 0; k < 3; ++k) {
            uniform(px, 4 * k, kG + k);
            for (uint32_t j = 1; j < 4; ++j) {
                uniform(px, 4 * k + j, kT0);
                sadd(kG + k, kT0, kG + k);
            }
            ssub_c(kG + k, kTwo, kG + k);
        }
        dot(kG, kG, kT1);
        ssqrt(kT1, kT2);
        for (uint32_t k = 0; k < 3; ++k) {
            sdiv(kG + k, kT2, kG + k);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kT1);
        imm(L1, kMinG2);
        TTI_SFPGT(0, L0, L1, kSetCC);  // 1e-12 > g2
        for (uint32_t k = 0; k < 3; ++k) {
            copy(n + k, kG + k);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        for (uint32_t k = 0; k < 3; ++k) {
            sadd(n + k, kG + k, kB + k);
        }
        dot(kB, kB, kT1);
        ssqrt(kT1, kT2);
        for (uint32_t k = 0; k < 3; ++k) {
            sdiv(kB + k, kT2, db + k);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, kT1);
        imm(L1, kMinB2);
        TTI_SFPGT(0, L0, L1, kSetCC);  // 1e-6 > b2
        for (uint32_t k = 0; k < 3; ++k) {
            copy(n + k, db + k);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // A packet's ray slots (o, d, inv, best at pk) for the lanes of mask (sign set):
    // origin o, direction d, best tmax; other lanes zero with best -1.
    inline void ray_out(
        uint32_t pk, uint32_t mask, uint32_t o, uint32_t d, uint32_t tmax
    ) {
        inverse_dir(d, kInv);
        for (uint32_t k = 0; k < 9; ++k) {
            fill(pk + k, 0);
        }
        fill(pk + kPkBest, kMinusOne);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(mask);
        for (uint32_t k = 0; k < 3; ++k) {
            copy(o + k, pk + kPkO + k);
            copy(d + k, pk + kPkD + k);
            copy(kInv + k, pk + kPkInv + k);
        }
        fill(pk + kPkBest, tmax);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // float32 bits of an integer below 2^24 (exact).
    inline uint32_t float_bits(uint32_t v) {
        if (v == 0) {
            return 0;
        }
        const uint32_t e = 31 - __builtin_clz(v);
        return ((127 + e) << 23) | ((v << (23 - e)) & 0x7FFFFF);
    }

    // Slot d = the integer c plus the slot a as integers (pixel numbers).
    inline void iadd_c(uint32_t a, uint32_t c, uint32_t d) {
        load(L0, a);
        imm(L1, c);
        TT_SFPIADD(0, L1, L0, kCcNone);
        store(L0, d);
    }

    // Slot m = -1.0 where slot s is not negative (a hit index), else 0.
    inline void flag_not_negative(uint32_t s, uint32_t m) {
        fill(m, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L0, s);
        TTI_SFPSETCC(0, L0, 0, kGte0);
        fill(m, kMinusOne);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // d = max(cos, 0), 0 where the shadow ray's best (slot occ) is negative: blocked.
    inline void direct(uint32_t cs, uint32_t occ, uint32_t d) {
        max0(cs, d);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(occ);
        fill(d, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // GEN: the primary rays of the packet whose lane 0 is pixel (x0, y0) into a packet
    // at kOut (ref camera_rays(), inverse_direction()); cn: the constant page.
    inline void gen(const uint32_t* cn, uint32_t x0, uint32_t y0) {
        const uint32_t* cam = cn + kCnCamera;
        fill(kT1, cam[12]);
        sadd_c(kRamp, float_bits(x0), kT0);  // x = float(px) + 0.5
        sadd_c(kT0, kHalf, kT0);
        ssub_c(kT0, cam[13], kT0);
        sdiv(kT0, kT1, kP);  // sx
        sadd_c(kRamp + 1, float_bits(y0), kT0);
        sadd_c(kT0, kHalf, kT0);
        ssub_c(kT0, cam[14], kT0);
        sdiv(kT0, kT1, kP + 1);  // sy
        for (uint32_t k = 0; k < 3; ++k) {
            smul_c(kP, cam[3 + k], kT0);  // (fwd + right sx) - up sy
            sadd_c(kT0, cam[9 + k], kT0);
            smul_c(kP + 1, cam[6 + k], kT2);
            ssub(kT0, kT2, kOut + kPkD + k);
            fill(kOut + kPkO + k, cam[k]);
        }
        inverse_dir(kOut + kPkD, kOut + kPkInv);
        fill(kOut + kPkBest, kTmax);
    }

    // SHADE1: primary packet at kIn, its hits at kIn + kPkVectors; out the shadow and
    // bounce packets and state 1 (ref render_pixels(), up to the first traces).
    inline void shade1(const uint32_t* cn, uint32_t pixel) {
        constexpr uint32_t pk = kIn, hit = kIn + kPkVectors;
        constexpr uint32_t shadow = kOut, bounce = kOut + kPkVectors;
        constexpr uint32_t st = kOut + 2 * kPkVectors;
        flag_not_negative(hit + kHitTri, kHitM);
        sky(pk + kPkD, st + kS1Col, cn);
        hit_point(pk + kPkO, pk + kPkD, hit + kHitT, kP);
        surface(
            hit + kHitN, hit + kHitCol, hit + kHitU, hit + kHitV, pk + kPkD, kN, kAlb
        );
        light_ray(kP, kN, kO2, kTl, st + kS1Cos, cn);
        iadd_c(kRamp + 2, pixel, kPx);
        bounce_dir(kPx, kN, kDb);
        ray_out(shadow, kHitM, kO2, kTl, kOne);
        ray_out(bounce, kHitM, kO2, kDb, kTmax);
        copy(kHitM, st + kS1Hit);
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(kHitM);
        for (uint32_t k = 0; k < 3; ++k) {
            copy(kAlb + k, st + kS1Col + k);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // SHADE2: state 1, the shadow hits, the bounce packet and its hits at kIn on; out
    // the shadow packet of the bounce hits and state 2.
    inline void shade2(const uint32_t* cn) {
        constexpr uint32_t s1 = kIn, occ = kIn + kS1Vectors;
        constexpr uint32_t pk = occ + kHitAnyVectors, hit = pk + kPkVectors;
        constexpr uint32_t shadow = kOut, s2 = kOut + kPkVectors;
        copy(s1 + kS1Hit, s2 + kS2Hit);
        for (uint32_t k = 0; k < 3; ++k) {
            copy(s1 + kS1Col + k, s2 + kS2Col + k);
        }
        direct(s1 + kS1Cos, occ + kHitT, kT3);
        sadd_c(kT3, kAmbient, s2 + kS2Light);
        flag_not_negative(hit + kHitTri, kHitM);
        copy(kHitM, s2 + kS2Hit1);
        sky(pk + kPkD, s2 + kS2Rad, cn);
        hit_point(pk + kPkO, pk + kPkD, hit + kHitT, kP);
        surface(
            hit + kHitN,
            hit + kHitCol,
            hit + kHitU,
            hit + kHitV,
            pk + kPkD,
            kN,
            s2 + kS2Alb1
        );
        light_ray(kP, kN, kO2, kTl, s2 + kS2Cos1, cn);
        ray_out(shadow, kHitM, kO2, kTl, kOne);
    }

    // IMAGE: state 2 and the second shadow hits at kIn; the colour at kOut.
    inline void image() {
        constexpr uint32_t s2 = kIn, occ = kIn + kS2Vectors;
        direct(s2 + kS2Cos1, occ + kHitT, kT3);
        sadd_c(kT3, kAmbient, kT3);
        for (uint32_t k = 0; k < 3; ++k) {
            copy(s2 + kS2Rad + k, kB + k);
        }
        TTI_SFPPUSHC(0, 0, 0, 0);
        where_negative(s2 + kS2Hit1);
        for (uint32_t k = 0; k < 3; ++k) {
            smul(s2 + kS2Alb1 + k, kT3, kB + k);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        for (uint32_t k = 0; k < 3; ++k) {
            smul_c(kB + k, kGi, kT0);
            sadd(s2 + kS2Light, kT0, kT0);
            copy(s2 + kS2Col + k, kOut + k);
            TTI_SFPPUSHC(0, 0, 0, 0);
            where_negative(s2 + kS2Hit);
            smul(s2 + kS2Col + k, kT0, kOut + k);
            TTI_SFPPOPC(0, 0, 0, 0);
        }
    }

}  // namespace shade
