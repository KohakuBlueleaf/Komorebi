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
    constexpr uint32_t kAddressMode = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kUShort = 2, kUpper = 8, kLower = 10;
    constexpr uint32_t kNegA = 1, kNegC = 2;                     // SFPMAD
    constexpr uint32_t kSetCC = 1;                               // SFPLE / SFPGT
    constexpr uint32_t kImmediate = 1, kCcNone = 4, kTwos = 2;   // SFPIADD
    constexpr uint32_t kLt0 = 0, kNe0 = 2, kGte0 = 4, kEq0 = 6;  // SFPSETCC
    constexpr uint32_t kScaleBits = 0x3F800003;                  // float32(1.0000004)
    constexpr uint32_t kInfBits = 0x7F800000;
    constexpr uint32_t kWallClock = 0xFFB121F0;
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    // Packet slots in Dst (a slot is 32 lanes): the ray, the results, the hit's normal
    // and colours, scratch, then the stack of entry distances (one slot per depth).
    constexpr uint32_t kSlotOrigin = 0;
    constexpr uint32_t kSlotDirection = 3;
    constexpr uint32_t kSlotInverse = 6;
    constexpr uint32_t kSlotBest = 9;
    constexpr uint32_t kSlotTriangle = 10;
    constexpr uint32_t kSlotU = 11;
    constexpr uint32_t kSlotV = 12;
    constexpr uint32_t kSlotNodes = 13;
    constexpr uint32_t kSlotLeaves = 14;
    constexpr uint32_t kSlotAttributes = 15;
    constexpr uint32_t kSlotPVector = 27;
    constexpr uint32_t kSlotDeterminant = 30;
    constexpr uint32_t kSlotInverseDeterminant = 31;
    constexpr uint32_t kSlotUNumerator = 32;
    constexpr uint32_t kSlotQVector = 33;
    constexpr uint32_t kSlotStack = 36;
    constexpr uint32_t kSlots = 256;

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    // SFPLOAD / SFPSTORE address of `slot`, and the window word of its lane 0 (lane
    // `lane` at + 2 * lane).
    constexpr uint32_t addr(uint32_t slot) {
        return 4 * (slot >> 1) + 2 * (slot & 1);
    }

    constexpr uint32_t word(uint32_t slot) {
        return 64 * (slot >> 1) + (slot & 1);
    }

    // Waits until earlier window stores are done (a load of the last word, used).
    inline void order(uint32_t last_word) {
        const uint32_t value = win()[last_word];
        asm volatile("and x0, x0, %0" : : "r"(value) : "memory");
    }

    // 32 words to `slot`, and back.
    inline void to_slot(uint32_t slot, const uint32_t* source) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        __riscv_vsse32_v_u32m8(
            const_cast<uint32_t*>(win() + word(slot)),
            8,
            __riscv_vle32_v_u32m8(source, vl),
            vl
        );
        asm volatile("" ::: "memory");
#else
        volatile uint32_t* window = win() + word(slot);
        for (uint32_t lane = 0; lane < 32; ++lane) {
            window[2 * lane] = source[lane];
        }
#endif
    }

    inline void from_slot(uint32_t slot, uint32_t* destination) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        __riscv_vse32_v_u32m8(
            destination,
            __riscv_vlse32_v_u32m8(
                const_cast<const uint32_t*>(win() + word(slot)), 8, vl
            ),
            vl
        );
        asm volatile("" ::: "memory");
#else
        const volatile uint32_t* window = win() + word(slot);
        for (uint32_t lane = 0; lane < 32; ++lane) {
            destination[lane] = window[2 * lane];
        }
#endif
    }

    // `words` words (a multiple of 32), contiguous on both sides (window or L1).
    inline void copy_words(
        volatile uint32_t* destination, const volatile uint32_t* source, uint32_t words
    ) {
#if defined(__riscv_vector)
        for (uint32_t i = 0; i < words; i += 32) {
            const size_t vl = __riscv_vsetvl_e32m8(32);
            __riscv_vse32_v_u32m8(
                const_cast<uint32_t*>(destination + i),
                __riscv_vle32_v_u32m8(const_cast<const uint32_t*>(source + i), vl),
                vl
            );
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t i = 0; i < words; ++i) {
            destination[i] = source[i];
        }
#endif
    }

    inline void fill_slot(uint32_t slot, uint32_t value) {
        volatile uint32_t* window = win() + word(slot);
        for (uint32_t lane = 0; lane < 32; ++lane) {
            window[2 * lane] = value;
        }
    }

    // A float32 (or integer) bit pattern into an LReg as two immediates.
    inline void immediate(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, kFloatB, bits >> 16);
        TT_SFPLOADI(lreg, kLower, bits & 0xFFFF);
    }

    inline void load(uint32_t lreg, uint32_t slot) {
        TT_SFPLOAD(lreg, kInt32, kAddressMode, addr(slot));
    }

    inline void store(uint32_t lreg, uint32_t slot) {
        TT_SFPSTORE(lreg, kInt32, kAddressMode, addr(slot));
    }

    // destination = left * right, left + right, left - right: one rounding each, as
    // separate float32 operations (the product's addend is -0, keeping a zero's sign).
    inline void mul(uint32_t left, uint32_t right, uint32_t destination) {
        TT_SFPMAD(left, right, ZERO, destination, kNegC);
    }

    inline void add(uint32_t left, uint32_t right, uint32_t destination) {
        TT_SFPMAD(left, ONE, right, destination, 0);
    }

    inline void sub(uint32_t left, uint32_t right, uint32_t destination) {
        TT_SFPMAD(left, ONE, right, destination, kNegC);
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
        immediate(L0, kScaleBits);
        TTI_SFPCONFIG(0, SCALE, 0);
        immediate(L0, kInfBits);
        TTI_SFPCONFIG(0, INF, 0);
        TTI_SFPNOP;
        tensix_sync();
    }

    inline void finish() {
        all_lanes();
        TTI_SFPNOP;
        tensix_sync();
    }

    // Slab test of the box words box(lreg, index) (min, max) for the enabled lanes:
    // L4 = entry, L5 = min(exit * scale + 0, best), L7 = entry where hit else +inf.
    template <typename Box>
    inline void box_test(Box box, uint32_t origin, uint32_t inverse, uint32_t best) {
        TTI_SFPLOADI(L4, kFloatB, 0);       // entry starts at +0
        TTI_SFPLOADI(L5, kFloatB, 0x7F80);  // exit at +inf
#pragma GCC unroll 3
        for (uint32_t axis = 0; axis < 3; ++axis) {
            box(L0, axis);
            box(L1, axis + 3);
            load(L2, origin + axis);
            load(L3, inverse + axis);
            sub(L0, L2, L0);  // (min - origin) and (max - origin)
            sub(L1, L2, L1);
            mul(L0, L3, L0);  // ... * inverse
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

    // reciprocal = 1 / value rounded to nearest (|value| in [2^-126, 2^103)): the
    // approximate reciprocal, two Newton steps, then an exact integer rounding test.
    inline void recip(
        uint32_t value,
        uint32_t reciprocal,
        uint32_t scratch0,
        uint32_t scratch1,
        uint32_t scratch2,
        uint32_t scratch3
    ) {
        TT_SFPARECIP(0, value, reciprocal, 0);
        // error = 1 - value * reciprocal; reciprocal += reciprocal * error
        TT_SFPMAD(value, reciprocal, ONE, scratch0, kNegA);
        TT_SFPMAD(reciprocal, scratch0, reciprocal, reciprocal, 0);
        TT_SFPMAD(value, reciprocal, ONE, scratch0, kNegA);
        TT_SFPMAD(reciprocal, scratch0, reciprocal, reciprocal, 0);
        // A, Y: significands of value, reciprocal. R = 2^47 - Y A, known mod 2^26 from
        // the low product bits; scratch2 = -64 R; reciprocal moves one where |2 R| > A.
        TT_SFPEXMAN(0, value, scratch0, 0);
        TT_SFPEXMAN(0, reciprocal, scratch1, 0);
        // high 23 bits of the low 23-bit product
        TT_SFPMUL24(scratch0, scratch1, ZERO, scratch2, 1);
        TT_SFPMUL24(scratch0, scratch1, ZERO, scratch3, 0);  // its low 23 bits
        TTI_SFPNOP;
        TT_SFPIADD(0, scratch1, scratch2, kCcNone);
        TT_SFPIADD(0, scratch0, scratch2, kCcNone);
        TT_SFPSHFT(29, 0, scratch2, 1);
        TT_SFPSHFT(6, 0, scratch3, 1);
        TT_SFPIADD(0, scratch3, scratch2, kCcNone);
        TT_SFPSHFT(5, 0, scratch0, 1);  // 32 A
        TT_SFPMOV(0, scratch2, scratch3, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, scratch0, scratch2, kLt0);  // 32 A - 64 R < 0: one up
        TT_SFPIADD(1, reciprocal, reciprocal, kImmediate | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, scratch0, scratch3, kTwos | kLt0);  // 32 A + 64 R < 0: one down
        TT_SFPIADD(0xFFF, reciprocal, reciprocal, kImmediate | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // The hit's attribute words (triangle words kTriAttr on) to slots kSlotAttributes
    // on, for the enabled lanes. Uses L5.
    template <typename Triangle>
    inline void store_attr(Triangle triangle) {
#pragma GCC unroll 12
        for (uint32_t attribute = 0; attribute < kTriAttrWords; ++attribute) {
            triangle(L5, kTriAttr + attribute);
            store(L5, kSlotAttributes + attribute);
        }
    }

    // Triangle test of words triangle(lreg, index), updating best, hit_index, u, v (and
    // attributes); any: a hit sets best = -1. Leaves L0 = v, L1 = t, L3 = u.
    template <typename Triangle>
    inline void tri_test(
        Triangle triangle,
        uint32_t origin,
        uint32_t direction,
        uint32_t best,
        uint32_t hit_index,
        uint32_t u,
        uint32_t v,
        bool any,
        bool with_attributes = false
    ) {
        // pv = cross(direction, e2)
        triangle(L0, 6);
        triangle(L1, 7);
        triangle(L2, 8);
        load(L3, direction);
        load(L4, direction + 1);
        load(L5, direction + 2);
        mul(L4, L2, L6);
        mul(L5, L1, L7);
        sub(L6, L7, L6);
        store(L6, kSlotPVector);
        mul(L5, L0, L6);
        mul(L3, L2, L7);
        sub(L6, L7, L6);
        store(L6, kSlotPVector + 1);
        mul(L3, L1, L6);
        mul(L4, L0, L7);
        sub(L6, L7, L6);
        store(L6, kSlotPVector + 2);
        // det = (e1 x pv x + e1 y pv y) + e1 z pv z; then 1 / det
        triangle(L0, 3);
        triangle(L1, 4);
        triangle(L2, 5);
        load(L3, kSlotPVector);
        load(L4, kSlotPVector + 1);
        mul(L0, L3, L3);
        mul(L1, L4, L4);
        add(L3, L4, L3);
        mul(L2, L6, L4);
        add(L3, L4, L3);
        store(L3, kSlotDeterminant);
        recip(L3, L4, L0, L1, L2, L5);
        store(L4, kSlotInverseDeterminant);
        // tv = origin - v0; u = dot(tv, pv) / det
        triangle(L0, 0);
        triangle(L1, 1);
        triangle(L2, 2);
        load(L3, origin);
        load(L4, origin + 1);
        load(L5, origin + 2);
        sub(L3, L0, L3);
        sub(L4, L1, L4);
        sub(L5, L2, L5);
        load(L0, kSlotPVector);
        mul(L3, L0, L0);
        load(L1, kSlotPVector + 1);
        mul(L4, L1, L1);
        add(L0, L1, L0);
        load(L1, kSlotPVector + 2);
        mul(L5, L1, L1);
        add(L0, L1, L0);
        load(L1, kSlotInverseDeterminant);
        mul(L0, L1, L0);
        store(L0, kSlotUNumerator);
        // qv = cross(tv, e1)
        triangle(L0, 3);
        triangle(L1, 4);
        triangle(L2, 5);
        mul(L4, L2, L6);
        mul(L5, L1, L7);
        sub(L6, L7, L6);
        store(L6, kSlotQVector);
        mul(L5, L0, L6);
        mul(L3, L2, L7);
        sub(L6, L7, L6);
        store(L6, kSlotQVector + 1);
        mul(L3, L1, L6);
        mul(L4, L0, L7);
        sub(L6, L7, L6);
        // v = dot(direction, qv) / det
        load(L0, direction);
        load(L1, kSlotQVector);
        mul(L0, L1, L0);
        load(L1, direction + 1);
        load(L2, kSlotQVector + 1);
        mul(L1, L2, L1);
        add(L0, L1, L0);
        load(L1, direction + 2);
        mul(L1, L6, L1);
        add(L0, L1, L0);
        load(L1, kSlotInverseDeterminant);
        mul(L0, L1, L0);
        // t = dot(e2, qv) / det
        triangle(L1, 6);
        load(L2, kSlotQVector);
        mul(L1, L2, L1);
        triangle(L2, 7);
        load(L3, kSlotQVector + 1);
        mul(L2, L3, L2);
        add(L1, L2, L1);
        triangle(L2, 8);
        mul(L2, L6, L2);
        add(L1, L2, L1);
        load(L2, kSlotInverseDeterminant);
        mul(L1, L2, L1);
        // ok: det != 0, u >= 0, u <= 1, v >= 0, u + v <= 1, t > 0 (zeros made +0)
        TTI_SFPPUSHC(0, 0, 0, 0);
        load(L2, kSlotDeterminant);
        TTI_SFPEXEXP(0, L2, L2, 1);
        TTI_SFPSETCC(0, L2, 0, kNe0);
        load(L3, kSlotUNumerator);
        TTI_SFPMAD(L3, ONE, ZERO, L4, 0);
        TTI_SFPSETCC(0, L4, 0, kGte0);
        TTI_SFPLE(0, ONE, L4, kSetCC);
        TTI_SFPMAD(L0, ONE, ZERO, L5, 0);
        TTI_SFPSETCC(0, L5, 0, kGte0);
        add(L3, L0, L5);
        TTI_SFPLE(0, ONE, L5, kSetCC);
        TTI_SFPGT(0, ZERO, L1, kSetCC);
        load(L2, best);
        triangle(L4, kTriIndex);
        if (any) {
            TTI_SFPGT(0, L1, L2, kSetCC);  // best > t
            TT_SFPSTORE(L4, kInt32, kAddressMode, addr(hit_index));
            TTI_SFPLOADI(L5, kFloatB, 0xBF80);
            TT_SFPSTORE(L5, kInt32, kAddressMode, addr(best));
        } else {
            TTI_SFPPUSHC(0, 0, 0, 0);
            TTI_SFPGT(0, L1, L2, kSetCC);  // best > t: closer
            TT_SFPSTORE(L1, kInt32, kAddressMode, addr(best));
            TT_SFPSTORE(L4, kInt32, kAddressMode, addr(hit_index));
            TT_SFPSTORE(L3, kInt32, kAddressMode, addr(u));
            TT_SFPSTORE(L0, kInt32, kAddressMode, addr(v));
            if (with_attributes) {
                store_attr(triangle);
            }
            TTI_SFPPOPC(0, 0, 0, 0);
            TTI_SFPPUSHC(0, 0, 0, 0);
            TT_SFPIADD(0, L1, L2, kTwos | kCcNone);  // t - best (bits)
            TTI_SFPSETCC(0, L2, 0, kEq0);
            TT_SFPLOAD(L5, kInt32, kAddressMode, addr(hit_index));
            TTI_SFPGT(0, L4, L5, kSetCC);  // current index > this one (-1: none)
            TT_SFPSTORE(L4, kInt32, kAddressMode, addr(hit_index));
            TT_SFPSTORE(L3, kInt32, kAddressMode, addr(u));
            TT_SFPSTORE(L0, kInt32, kAddressMode, addr(v));
            if (with_attributes) {
                store_attr(triangle);
            }
            TTI_SFPPOPC(0, 0, 0, 0);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // Box-test and triangle-test counters: `slot` += `amount` on the enabled lanes.
    inline void count(uint32_t slot, uint32_t amount) {
        load(L6, slot);
        TT_SFPIADD(amount, L6, L6, kImmediate | kCcNone);
        store(L6, slot);
    }

}  // namespace rt
