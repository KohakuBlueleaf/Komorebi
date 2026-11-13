// The vector unit's part of COUNT and SETUP, issued by the pack thread as raw Tensix
// instructions: camera space, screen, box and records for 32 triangles, in the
// reference's order of float32 operations (ref/raster.py).
#pragma once

#include <stdint.h>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#include "setup.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace su {

    using namespace ckernel;

    // LRegs: L0 to L7 work registers; 9 = 0, 10 = 1.0.
    constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
    constexpr uint32_t ZERO = 9, ONE = 10;
    constexpr uint32_t kInt32 = 4;  // SFPLOAD / SFPSTORE: 32 bits as they are
    constexpr uint32_t kAm = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kLower = 10;
    constexpr uint32_t kNegA = 1, kNegC = 2;  // SFPMAD
    constexpr uint32_t kSetCC = 1;            // SFPGT
    constexpr uint32_t kSign = 1;             // SFPMOV: the sign inverted
    constexpr uint32_t kCcNone = 4;           // SFPIADD
    constexpr uint32_t kLt0 = 0;              // SFPSETCC
    constexpr uint32_t kWallClock = 0xFFB121F0;
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    // Dst slots (32 lanes each): the block (kBlockSlots), the camera's 16 numbers
    // copied to every lane, camera space x, y, z of each corner, the records (slot
    // kSRec + w is word w), screen x, y, 1/z of each corner, the area and 1/area.
    constexpr uint32_t kSCol = 9;
    constexpr uint32_t kSCam = 32;
    constexpr uint32_t kSCs = 48;
    constexpr uint32_t kSRec = 64;
    constexpr uint32_t kSScr = 96;
    constexpr uint32_t kSArea = 106;
    constexpr uint32_t kSInv = 107;

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

    // Waits until earlier window stores are done (a load of word w, used).
    inline void order(uint32_t w) {
        const uint32_t v = win()[w];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
    }

    inline void fill_slot(uint32_t s, uint32_t v) {
        volatile uint32_t* w = win() + word(s);
        for (uint32_t l = 0; l < 32; ++l) {
            w[2 * l] = v;
        }
    }

    // A block of kBlockSlots vectors, already in the window's order, to slots 0 on.
    inline void block_in(const uint32_t* src) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m8(32);
        for (uint32_t i = 0; i < kBlockSlots * 32; i += 32) {
            __riscv_vse32_v_u32m8(
                const_cast<uint32_t*>(win() + i), __riscv_vle32_v_u32m8(src + i, vl), vl
            );
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t i = 0; i < kBlockSlots * 32; ++i) {
            win()[i] = src[i];
        }
#endif
        order(kBlockSlots * 32 - 1);
    }

    // The 32 record slots to 32 records of 128 bytes (word w of record l: slot kSRec +
    // w, lane l), two slots per segment load of 16 lanes (the register group's size).
    inline void records_out(uint32_t* rec) {
#if defined(__riscv_vector)
        const size_t vl = __riscv_vsetvl_e32m4(16);
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t k = 0; k < 16; ++k) {
                const auto v = (
                    __riscv_vlseg2e32_v_u32m4x2(
                        const_cast<const uint32_t*>(win() + word(kSRec) + 64 * k + 32 * h),
                        vl
                    )
                );
                __riscv_vssseg2e32_v_u32m4x2(rec + 512 * h + 2 * k, 128, v, vl);
            }
        }
        asm volatile("" ::: "memory");
#else
        for (uint32_t w = 0; w < 32; ++w) {
            for (uint32_t l = 0; l < 32; ++l) {
                rec[32 * l + w] = win()[word(kSRec + w) + 2 * l];
            }
        }
#endif
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

    // d = -a, the sign bit inverted (as numpy's negation).
    inline void neg(uint32_t a, uint32_t d) {
        TT_SFPMOV(0, a, d, kSign);
    }

    // hi = max(hi, lo), lo = min(hi, lo).
    inline void order_pair(uint32_t hi, uint32_t lo) {
        TTI_SFPNOP;
        TT_SFPSWAP(0, hi, lo, 1);
    }

    // Thread state: no address step, Dst offset 0, 32-bit Dst, every lane enabled.
    inline void setup() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7
        );
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        tensix_sync();
    }

    inline void finish() {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        tensix_sync();
    }

    // y = 1 / a rounded to nearest (|a| in [2^-126, 2^103)): the approximate
    // reciprocal, two Newton steps, then the rounding fixed by an exact integer test
    // (the ray tracer's, projects/raytrace/kernels/sfpu.hpp).
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
        TT_SFPIADD(1, y, y, 1 | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
        TTI_SFPPUSHC(0, 0, 0, 0);
        TT_SFPIADD(0, t0, t3, 2 | kLt0);  // 32 A + 64 R < 0: one down
        TT_SFPIADD(0xFFF, y, y, 1 | kCcNone);
        TTI_SFPPOPC(0, 0, 0, 0);
    }

    // Camera space of every corner: d = p - eye; x = (d0 c3 + d1 c4) + d2 c5, and y,
    // z with the next two rows of the camera.
    inline void camera_space() {
        for (uint32_t k = 0; k < 3; ++k) {
            for (uint32_t i = 0; i < 3; ++i) {
                load(L0, 3 * k + i);
                load(L1, kSCam + i);
                sub(L0, L1, L2 + i);
            }
            for (uint32_t j = 0; j < 3; ++j) {
                load(L5, kSCam + 3 + 3 * j);
                load(L6, kSCam + 4 + 3 * j);
                load(L7, kSCam + 5 + 3 * j);
                mul(L2, L5, L0);
                mul(L3, L6, L1);
                add(L0, L1, L0);
                mul(L4, L7, L1);
                add(L0, L1, L0);
                store(L0, kSCs + 3 * k + j);
            }
        }
    }

    // Screen x, y and 1/z of every corner: inv = 1 / z, x = c13 + c12 (x inv), y = c14
    // - c12 (y inv). Lanes with a corner short of the near distance get numbers that
    // are not used.
    inline void screen() {
        for (uint32_t k = 0; k < 3; ++k) {
            load(L0, kSCs + 3 * k + 2);
            recip(L0, L1, L2, L3, L4, L5);
            store(L1, kSScr + 3 * k + 2);
            load(L2, kSCam + 12);
            load(L0, kSCs + 3 * k);
            mul(L0, L1, L0);
            mul(L2, L0, L0);
            load(L3, kSCam + 13);
            add(L3, L0, L0);
            store(L0, kSScr + 3 * k);
            load(L0, kSCs + 3 * k + 1);
            mul(L0, L1, L0);
            mul(L2, L0, L0);
            load(L3, kSCam + 14);
            sub(L3, L0, L0);
            store(L0, kSScr + 3 * k + 1);
        }
    }

    // ceil or floor of L0 (clamped to [lo, hi] first) as 1.5 x 2^23 + n, to slot s.
    inline void round_box(uint32_t lo, uint32_t hi, bool up, uint32_t s) {
        imm(L3, lo);
        order_pair(L0, L3);
        imm(L3, hi);
        order_pair(L3, L0);
        imm(L4, kMagic);
        add(L0, L4, L5);  // to the nearest integer
        sub(L5, L4, L6);
        TTI_SFPNOP;
        TTI_SFPPUSHC(0, 0, 0, 0);
        if (up) {
            TTI_SFPGT(0, L6, L0, kSetCC);  // rounded below the value: one up
            add(L5, ONE, L5);
        } else {
            TTI_SFPGT(0, L0, L6, kSetCC);  // rounded above the value: one down
            sub(L5, ONE, L5);
        }
        TTI_SFPPOPC(0, 0, 0, 0);
        store(L5, s);
    }

    // The box of coordinate c (0: x, 1: y) of the three corners, to record words
    // kRawX0 + 2 c and kRawX1 + 2 c: ceil(lo - 0.5) in [0, size], floor(hi - 0.5) in
    // [-1, size - 1].
    inline void box(uint32_t c, uint32_t size_bits, uint32_t last_bits) {
        load(L0, kSScr + c);
        load(L1, kSScr + 3 + c);
        load(L2, kSScr + 6 + c);
        order_pair(L1, L0);
        order_pair(L2, L0);
        order_pair(L1, L2);
        imm(L7, 0x3F000000);  // 0.5
        sub(L0, L7, L0);
        sub(L1, L7, L2);
        round_box(0xBF000000, size_bits, true, kSRec + kRawX0 + 2 * c);
        TT_SFPMOV(0, L2, L0, 0);
        round_box(0xBF800000, last_bits, false, kSRec + kRawX1 + 2 * c);
    }

    // The doubled area, raw to kRawArea; corners 1 and 2 swapped where it is
    // negative (their screen numbers and colours), the area then made positive.
    inline void area_and_order() {
        load(L0, kSScr + 3);
        load(L1, kSScr);
        sub(L0, L1, L2);  // x1 - x0
        load(L0, kSScr + 7);
        load(L3, kSScr + 1);
        sub(L0, L3, L4);  // y2 - y0
        mul(L2, L4, L2);
        load(L0, kSScr + 6);
        sub(L0, L1, L4);  // x2 - x0
        load(L0, kSScr + 4);
        sub(L0, L3, L5);  // y1 - y0
        mul(L4, L5, L4);
        sub(L2, L4, L2);
        store(L2, kSRec + kRawArea);
        TTI_SFPNOP;
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPSETCC(0, L2, 0, kLt0);
        for (uint32_t j = 0; j < 3; ++j) {
            load(L0, kSScr + 3 + j);
            load(L1, kSScr + 6 + j);
            store(L1, kSScr + 3 + j);
            store(L0, kSScr + 6 + j);
            load(L0, kSCol + 3 + j);
            load(L1, kSCol + 6 + j);
            store(L1, kSCol + 3 + j);
            store(L0, kSCol + 6 + j);
        }
        neg(L2, L2);
        TTI_SFPPOPC(0, 0, 0, 0);
        store(L2, kSArea);
    }

    // Edge k between corners p and q: a = y_p - y_q, b = x_q - x_p, c = -(a x_p + b
    // y_p), to record words k, 3 + k, 6 + k.
    inline void edges() {
        constexpr uint32_t pq[3][2] = {{1, 2}, {2, 0}, {0, 1}};
        for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t p = kSScr + 3 * pq[k][0], q = kSScr + 3 * pq[k][1];
            load(L0, p);
            load(L1, p + 1);
            load(L2, q);
            load(L3, q + 1);
            sub(L1, L3, L4);  // a
            sub(L2, L0, L5);  // b
            store(L4, kSRec + k);
            store(L5, kSRec + 3 + k);
            mul(L4, L0, L4);
            mul(L5, L1, L5);
            add(L4, L5, L4);
            TTI_SFPNOP;
            neg(L4, L4);
            store(L4, kSRec + 6 + k);
        }
    }

    // The plane of per-corner values in slots v, v + step, v + 2 step: for each of the
    // a, b, c coefficients, ((c0 v0 + c1 v1) + c2 v2) / area, to record words w + k.
    inline void plane(uint32_t v, uint32_t step, uint32_t w) {
        for (uint32_t k = 0; k < 3; ++k) {
            load(L0, kSRec + 3 * k);
            load(L1, v);
            mul(L0, L1, L2);
            load(L0, kSRec + 3 * k + 1);
            load(L1, v + step);
            mul(L0, L1, L3);
            add(L2, L3, L2);
            load(L0, kSRec + 3 * k + 2);
            load(L1, v + 2 * step);
            mul(L0, L1, L3);
            add(L2, L3, L2);
            load(L4, kSInv);
            mul(L2, L4, L2);
            store(L2, kSRec + w + k);
        }
    }

    // Everything of a block whose corners' camera space is in place: screen, area,
    // box, the camera distances, and with `planes` the edges and planes.
    inline void triangles(bool planes) {
        screen();
        area_and_order();
        box(0, 0x44800000, 0x447FC000);  // 1024, 1023
        box(1, 0x44800000, 0x447FC000);
        for (uint32_t k = 0; k < 3; ++k) {
            load(L0, kSCs + 3 * k + 2);
            store(L0, kSRec + kRawZ + k);
        }
        if (!planes) {
            return;
        }
        edges();
        load(L0, kSArea);
        recip(L0, L1, L2, L3, L4, L5);
        store(L1, kSInv);
        plane(kSScr + 2, 3, 12);
        for (uint32_t ch = 0; ch < 3; ++ch) {
            plane(kSCol + ch, 3, 15 + 3 * ch);
        }
    }

}  // namespace su
