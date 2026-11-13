// The vector-unit side of the ray walk, submitted by the pack thread as raw Tensix
// instructions: the reference's slab test and triangle test for 32 rays, bit for bit.
#pragma once

#include <stdint.h>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#include "raytrace.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rt {

    using namespace ckernel;

    // LRegs: L0 to L7 work registers; 9 = 0, 10 = 1.0; 11 = the slab scale, 12 = +inf
    // (set by setup()); 15 = 2 lane.
    constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
    constexpr uint32_t ZERO = 9, ONE = 10, SCALE = 11, INF = 12, LANE = 15;
    constexpr uint32_t kInt32 = 4;  // SFPLOAD / SFPSTORE: 32 bits as they are
    constexpr uint32_t kAm = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kUShort = 2, kUpper = 8, kLower = 10;
    constexpr uint32_t kNegA = 1, kNegC = 2;                     // SFPMAD
    constexpr uint32_t kSetCC = 1;                               // SFPLE / SFPGT
    constexpr uint32_t kImm = 1, kCcNone = 4, kTwos = 2;         // SFPIADD
    constexpr uint32_t kLt0 = 0, kNe0 = 2, kGte0 = 4, kEq0 = 6;  // SFPSETCC
    constexpr uint32_t kScaleBits = 0x3F800003;                  // float32(1.0000004)
    constexpr uint32_t kInfBits = 0x7F800000;
    constexpr uint32_t kWallClock = 0xFFB121F0;
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    // Packet slots in Dst (a slot is 32 lanes): the ray, the results, the hit's normal
    // and colours, scratch, then the stack of entry distances (one slot per depth).
    constexpr uint32_t kSO = 0;
    constexpr uint32_t kSD = 3;
    constexpr uint32_t kSInv = 6;
    constexpr uint32_t kSBest = 9;
    constexpr uint32_t kSTri = 10;
    constexpr uint32_t kSU = 11;
    constexpr uint32_t kSV = 12;
    constexpr uint32_t kSNodes = 13;
    constexpr uint32_t kSLeaves = 14;
    constexpr uint32_t kSAttr = 15;
    constexpr uint32_t kSPv = 27, kSDet = 30, kSDetInv = 31, kSUt = 32, kSQ = 33;
    constexpr uint32_t kSStack = 36;
    constexpr uint32_t kSlots = 256;

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    // SFPLOAD / SFPSTORE address of slot s, and the window word of its lane 0 (lane l
    // at + 2 l).
    constexpr uint32_t addr(uint32_t s) {
        return 4 * (s >> 1) + 2 * (s & 1);
    }

    constexpr uint32_t word(uint32_t s) {
        return 64 * (s >> 1) + (s & 1);
    }

    // Waits until earlier window stores are done (a load of the last word, used).
    inline void order(uint32_t w) {
        const uint32_t v = win()[w];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
    }

    // 32 words to slot s, and back.
    inline void to_slot(uint32_t s, const uint32_t* src) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        __riscv_vsse32_v_u32m8(
            const_cast<uint32_t*>(win() + word(s)),
            8,
            __riscv_vle32_v_u32m8(src, vl),
            vl
        );
        asm volatile("" ::: "memory");
#else
        volatile uint32_t* w = win() + word(s);
        for (uint32_t l = 0; l < 32; ++l) {
            w[2 * l] = src[l];
        }
#endif
    }

    inline void from_slot(uint32_t s, uint32_t* dst) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        __riscv_vse32_v_u32m8(
            dst,
            __riscv_vlse32_v_u32m8(const_cast<const uint32_t*>(win() + word(s)), 8, vl),
            vl
        );
        asm volatile("" ::: "memory");
#else
        const volatile uint32_t* w = win() + word(s);
        for (uint32_t l = 0; l < 32; ++l) {
            dst[l] = w[2 * l];
        }
#endif
    }

    // n words (a multiple of 32), contiguous on both sides (window or L1).
    inline void copy_words(
        volatile uint32_t* dst, const volatile uint32_t* src, uint32_t n
    ) {
#if defined(__riscv_vector)
        for (uint32_t i = 0; i < n; i += 32) {
            const size_t vl = __riscv_vsetvl_e32m8(32);
            __riscv_vse32_v_u32m8(
                const_cast<uint32_t*>(dst + i),
                __riscv_vle32_v_u32m8(const_cast<const uint32_t*>(src + i), vl),
                vl
            );
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
#endif
    }

    inline void fill_slot(uint32_t s, uint32_t v) {
        volatile uint32_t* w = win() + word(s);
        for (uint32_t l = 0; l < 32; ++l) {
            w[2 * l] = v;
        }
    }

    // A float32 (or integer) bit pattern into an LReg as two immediates.
    inline void imm(uint32_t r, uint32_t bits) {
        TT_SFPLOADI(r, kFloatB, bits >> 16);
        TT_SFPLOADI(r, kLower, bits & 0xFFFF);
    }

    inline void load(uint32_t r, uint32_t s) {
        TT_SFPLOAD(r, kInt32, kAm, addr(s));
    }

    inline void store(uint32_t r, uint32_t s) {
        TT_SFPSTORE(r, kInt32, kAm, addr(s));
    }

    // d = a * b, d = a + b, d = a - b: one rounding each, as separate float32
    // operations (the product's addend is -0, which keeps the sign of a zero product).
    inline void mul(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, b, ZERO, d, kNegC);
    }

    inline void add(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, ONE, b, d, 0);
    }

    inline void sub(uint32_t a, uint32_t b, uint32_t d) {
        TT_SFPMAD(a, ONE, b, d, kNegC);
    }

    // Every lane enabled, predication on.
    inline void all_lanes() {
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // Thread state: no address step, Dst offset 0, 32-bit Dst, every lane enabled; the
    // slab scale and +inf into LReg 11 and 12.
    inline void setup() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7
        );
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
        all_lanes();
        imm(L0, kScaleBits);
        TTI_SFPCONFIG(0, SCALE, 0);
        imm(L0, kInfBits);
        TTI_SFPCONFIG(0, INF, 0);
        TTI_SFPNOP;
        tensix_sync();
    }

    inline void finish() {
        all_lanes();
        TTI_SFPNOP;
        tensix_sync();
    }

    // Slab test of the box with words box(r, k) (min, max) for the enabled lanes: L4 =
    // entry, L5 = min(exit * scale + 0, best), L7 = entry where hit else +inf.
    template <typename Box>
    inline void box_test(Box box, uint32_t o, uint32_t inv, uint32_t best) {
        TTI_SFPLOADI(L4, kFloatB, 0);       // entry starts at +0
        TTI_SFPLOADI(L5, kFloatB, 0x7F80);  // exit at +inf
#pragma GCC unroll 3
        for (uint32_t k = 0; k < 3; ++k) {
            box(L0, k);
            box(L1, k + 3);
            load(L2, o + k);
            load(L3, inv + k);
            sub(L0, L2, L0);  // (min - o) and (max - o)
            sub(L1, L2, L1);
            mul(L0, L3, L0);  // ... * inv
            mul(L1, L3, L1);
            TTI_SFPNOP;
            TTI_SFPSWAP(0, L1, L0, 1);  // L0 near, L1 far
            TTI_SFPSWAP(0, L4, L0, 1);  // L4 = max(L4, near)
            TTI_SFPSWAP(0, L1, L5, 1);  // L5 = min(L5, far)
        }
        TTI_SFPMAD(L5, SCALE, ZERO, L5, 0);  // exit * scale + 0: -0 becomes +0
        load(L6, best);
        TTI_SFPNOP;
        TTI_SFPSWAP(0, L6, L5, 1);  // L5 = min(exit, best)
        TTI_SFPMOV(0, INF, L7, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPLE(0, L5, L4, kSetCC);  // hit: entry <= min(exit, best)
        TTI_SFPMOV(0, L4, L7, 0);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // y = 1 / a rounded to nearest (|a| in [2^-126, 2^103)): the approximate
    // reciprocal, two Newton steps, then the rounding fixed by an exact integer test.
    inline void recip(
        uint32_t a, uint32_t y, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3
    ) {
        TT_SFPARECIP(0, a, y, 0);
        TT_SFPMAD(a, y, ONE, t0, kNegA);  // e = 1 - a y
        TT_SFPMAD(y, t0, y, y, 0);        // y = y e + y
        TT_SFPMAD(a, y, ONE, t0, kNegA);
        TT_SFPMAD(y, t0, y, y, 0);
        // A, Y: significands of a and y. R = 2^47 - Y A, known mod 2^26 from the low
        // product bits; t2 = -64 R, then y moves by one where |2 R| > A.
        TT_SFPEXMAN(0, a, t0, 0);
        TT_SFPEXMAN(0, y, t1, 0);
        TT_SFPMUL24(t0, t1, ZERO, t2, 1);  // high 23 bits of the low 23-bit product
        TT_SFPMUL24(t0, t1, ZERO, t3, 0);  // its low 23 bits
        TTI_SFPNOP;
        TT_SFPIADD(0, t1, t2, kCcNone);
        TT_SFPIADD(0, t0, t2, kCcNone);
        TT_SFPSHFT(29, 0, t2, 1);
        TT_SFPSHFT(6, 0, t3, 1);
        TT_SFPIADD(0, t3, t2, kCcNone);
        TT_SFPSHFT(5, 0, t0, 1);  // 32 A
        TT_SFPMOV(0, t2, t3, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, t0, t2, kLt0);  // 32 A - 64 R < 0: one up
        TT_SFPIADD(1, y, y, kImm | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, t0, t3, kTwos | kLt0);  // 32 A + 64 R < 0: one down
        TT_SFPIADD(0xFFF, y, y, kImm | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // The hit's attribute words (tri words kTriAttr on) to slots kSAttr on, for the
    // enabled lanes. Uses L5.
    template <typename Tri>
    inline void store_attr(Tri tri) {
#pragma GCC unroll 12
        for (uint32_t k = 0; k < kTriAttrWords; ++k) {
            tri(L5, kTriAttr + k);
            store(L5, kSAttr + k);
        }
    }

    // Triangle test of words tri(r, k) and the update of best, tri, u, v (attr: and the
    // attributes); any: a hit sets best = -1. Leaves L0 = v, L1 = t, L3 = u.
    template <typename Tri>
    inline void tri_test(
        Tri tri,
        uint32_t o,
        uint32_t d,
        uint32_t best,
        uint32_t cur,
        uint32_t u,
        uint32_t v,
        bool any,
        bool attr = false
    ) {
        // pv = cross(d, e2)
        tri(L0, 6);
        tri(L1, 7);
        tri(L2, 8);
        load(L3, d);
        load(L4, d + 1);
        load(L5, d + 2);
        mul(L4, L2, L6);
        mul(L5, L1, L7);
        sub(L6, L7, L6);
        store(L6, kSPv);
        mul(L5, L0, L6);
        mul(L3, L2, L7);
        sub(L6, L7, L6);
        store(L6, kSPv + 1);
        mul(L3, L1, L6);
        mul(L4, L0, L7);
        sub(L6, L7, L6);
        store(L6, kSPv + 2);
        // det = (e1 x pv x + e1 y pv y) + e1 z pv z; then 1 / det
        tri(L0, 3);
        tri(L1, 4);
        tri(L2, 5);
        load(L3, kSPv);
        load(L4, kSPv + 1);
        mul(L0, L3, L3);
        mul(L1, L4, L4);
        add(L3, L4, L3);
        mul(L2, L6, L4);
        add(L3, L4, L3);
        store(L3, kSDet);
        recip(L3, L4, L0, L1, L2, L5);
        store(L4, kSDetInv);
        // tv = o - v0; u = dot(tv, pv) / det
        tri(L0, 0);
        tri(L1, 1);
        tri(L2, 2);
        load(L3, o);
        load(L4, o + 1);
        load(L5, o + 2);
        sub(L3, L0, L3);
        sub(L4, L1, L4);
        sub(L5, L2, L5);
        load(L0, kSPv);
        mul(L3, L0, L0);
        load(L1, kSPv + 1);
        mul(L4, L1, L1);
        add(L0, L1, L0);
        load(L1, kSPv + 2);
        mul(L5, L1, L1);
        add(L0, L1, L0);
        load(L1, kSDetInv);
        mul(L0, L1, L0);
        store(L0, kSUt);
        // qv = cross(tv, e1)
        tri(L0, 3);
        tri(L1, 4);
        tri(L2, 5);
        mul(L4, L2, L6);
        mul(L5, L1, L7);
        sub(L6, L7, L6);
        store(L6, kSQ);
        mul(L5, L0, L6);
        mul(L3, L2, L7);
        sub(L6, L7, L6);
        store(L6, kSQ + 1);
        mul(L3, L1, L6);
        mul(L4, L0, L7);
        sub(L6, L7, L6);
        // v = dot(d, qv) / det
        load(L0, d);
        load(L1, kSQ);
        mul(L0, L1, L0);
        load(L1, d + 1);
        load(L2, kSQ + 1);
        mul(L1, L2, L1);
        add(L0, L1, L0);
        load(L1, d + 2);
        mul(L1, L6, L1);
        add(L0, L1, L0);
        load(L1, kSDetInv);
        mul(L0, L1, L0);
        // t = dot(e2, qv) / det
        tri(L1, 6);
        load(L2, kSQ);
        mul(L1, L2, L1);
        tri(L2, 7);
        load(L3, kSQ + 1);
        mul(L2, L3, L2);
        add(L1, L2, L1);
        tri(L2, 8);
        mul(L2, L6, L2);
        add(L1, L2, L1);
        load(L2, kSDetInv);
        mul(L1, L2, L1);
        // ok: det != 0, u >= 0, u <= 1, v >= 0, u + v <= 1, t > 0 (zeros made +0)
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L2, kSDet);
        TTI_SFPEXEXP(0, L2, L2, 1);
        TTI_SFPSETCC(0, L2, 0, kNe0);
        load(L3, kSUt);
        TTI_SFPMAD(L3, ONE, ZERO, L4, 0);
        TTI_SFPSETCC(0, L4, 0, kGte0);
        TTI_SFPLE(0, ONE, L4, kSetCC);
        TTI_SFPMAD(L0, ONE, ZERO, L5, 0);
        TTI_SFPSETCC(0, L5, 0, kGte0);
        add(L3, L0, L5);
        TTI_SFPLE(0, ONE, L5, kSetCC);
        TTI_SFPGT(0, ZERO, L1, kSetCC);
        load(L2, best);
        tri(L4, kTriIndex);
        if (any) {
            TTI_SFPGT(0, L1, L2, kSetCC);  // best > t
            TT_SFPSTORE(L4, kInt32, kAm, addr(cur));
            TTI_SFPLOADI(L5, kFloatB, 0xBF80);
            TT_SFPSTORE(L5, kInt32, kAm, addr(best));
        } else {
            TTI_SFPPUSHC(0, 0, 0, 0);
            TTI_SFPGT(0, L1, L2, kSetCC);  // best > t: closer
            TT_SFPSTORE(L1, kInt32, kAm, addr(best));
            TT_SFPSTORE(L4, kInt32, kAm, addr(cur));
            TT_SFPSTORE(L3, kInt32, kAm, addr(u));
            TT_SFPSTORE(L0, kInt32, kAm, addr(v));
            if (attr) {
                store_attr(tri);
            }
            TTI_SFPPOPC(0, 0, 0, 0);
            TTI_SFPPUSHC(0, 0, 0, 0);
            TT_SFPIADD(0, L1, L2, kTwos | kCcNone);  // t - best (bits)
            TTI_SFPSETCC(0, L2, 0, kEq0);
            TT_SFPLOAD(L5, kInt32, kAm, addr(cur));
            TTI_SFPGT(0, L4, L5, kSetCC);  // current index > this one (-1: none)
            TT_SFPSTORE(L4, kInt32, kAm, addr(cur));
            TT_SFPSTORE(L3, kInt32, kAm, addr(u));
            TT_SFPSTORE(L0, kInt32, kAm, addr(v));
            if (attr) {
                store_attr(tri);
            }
            TTI_SFPPOPC(0, 0, 0, 0);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // Box-test and triangle-test counters: slot s += n on the enabled lanes.
    inline void count(uint32_t s, uint32_t n) {
        load(L6, s);
        TT_SFPIADD(n, L6, L6, kImm | kCcNone);
        store(L6, s);
    }

}  // namespace rt
