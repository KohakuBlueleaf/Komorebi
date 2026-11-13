// The pack thread's part of the rasterizer: it submits the vector-unit instructions and
// moves data between L1 and Dst through the Dst window. All raw Tensix instructions.
#pragma once

#include <stdint.h>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#include "raster.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace fill {

    using namespace ckernel;

    // LRegs: L0, L1 work; L2 to L4 edge thresholds, L5 triangle number (in rows); L6
    // = -0, L7 = py; 9 = +0, 10 = 1.0, 12 = -0's neighbour 0x80000001, 15 = 2 lane.
    constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, NZ = 6, PY = 7;
    constexpr uint32_t ZERO = 9, ONE = 10, TNEG = 12, LANE = 15;
    constexpr uint32_t kInt32 = 4;  // SFPLOAD / SFPSTORE: 32 bits as they are
    constexpr uint32_t kAm = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kFloatA = 1, kUShort = 2, kLower = 10;
    constexpr uint32_t kSetCC = 1;  // SFPGT: lanes stay enabled where VD > VC
    constexpr uint32_t kWallClock = 0xFFB121F0;
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    // Dst vectors: plane p row r at 32 p + r (depth, red, green, blue, triangle), py of
    // row r at 160 + r, the triangle's broadcast constants from 192.
    constexpr uint32_t kVecPy = 160;
    constexpr uint32_t kVecK = 192;
    constexpr uint32_t kKA = 0;
    constexpr uint32_t kKB = 3;
    constexpr uint32_t kKC = 6;
    constexpr uint32_t kKW = 9;
    constexpr uint32_t kKColA = 12;
    constexpr uint32_t kKColB = 15;
    constexpr uint32_t kKColC = 18;
    constexpr uint32_t kKPx = 21;

    // Dst address of vector v: an aligned group of four rows, even or odd columns.
    constexpr uint32_t addr(uint32_t v) {
        return 4 * (v >> 1) + 2 * (v & 1);
    }

    constexpr uint32_t kaddr(uint32_t k) {
        return addr(kVecK + k);
    }

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    // n words (a multiple of 8), loads grouped ahead of the stores that use them.
    inline void copy_words(
        volatile uint32_t* dst, const volatile uint32_t* src, uint32_t n
    ) {
        for (uint32_t i = 0; i < n; i += 8) {
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                v[j] = src[i + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                dst[i + j] = v[j];
            }
        }
    }

#if defined(__riscv_vector)
    // The same copy on the pack core's vector extension, 32 words per instruction.
    inline void copy_words_rvv(uint32_t* dst, const uint32_t* src, uint32_t n) {
        const size_t vl = __riscv_vsetvl_e32m8(32);
        for (uint32_t i = 0; i < n; i += 32) {
            __riscv_vse32_v_u32m8(dst + i, __riscv_vle32_v_u32m8(src + i, vl), vl);
        }
        asm volatile("" ::: "memory");
    }
#endif

    // Waits until earlier window stores are done (a load of the last word, used).
    inline void order(uint32_t word) {
        uint32_t v = win()[word];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
    }

    // Thread state: no address step, Dst offset 0, 32-bit window, every lane enabled;
    // L6 = -0, L12 = 0x80000001.
    inline void setup() {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(
            ADDR_MOD_7
        );
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPLOADI(NZ, kFloatB, 0x8000);
        TTI_SFPLOADI(L0, kFloatB, 0x8000);
        TTI_SFPLOADI(L0, kLower, 0x0001);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, TNEG, 0);
        TTI_SFPNOP;
        tensix_sync();
    }

    inline void finish() {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        tensix_sync();
    }

    // A float32 bit pattern into an LReg (two immediates; one when the low half is 0).
    inline void load_f32(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, kFloatB, bits >> 16);
        if (bits & 0xFFFF) {
            TT_SFPLOADI(lreg, kLower, bits & 0xFFFF);
        }
    }

    inline void load_vec(uint32_t lreg, uint32_t v) {
        TT_SFPLOAD(lreg, kInt32, kAm, addr(v));
    }

    inline void store_vec(uint32_t lreg, uint32_t v) {
        TT_SFPSTORE(lreg, kInt32, kAm, addr(v));
    }

    // Cycles of the pack thread, by phase (wall clock).
    struct Prof {
        uint32_t wait_in = 0;   // waiting for list pages
        uint32_t init = 0;      // planes, px and py of a tile
        uint32_t tri = 0;       // a triangle's constants
        uint32_t rows = 0;      // the row bodies
        uint32_t drain = 0;     // vector unit finishing before the read-out
        uint32_t out = 0;       // planes from Dst to the output pages
        uint32_t wait_out = 0;  // waiting for free output pages
        uint32_t tiles = 0;
        uint32_t tris = 0;
        uint32_t row_count = 0;
        uint32_t total = 0;
    };

    // Planes of tile (tx, ty) to their empty values; py of its rows and px of its
    // columns into Dst.
    inline void init_tile(uint32_t tx, uint32_t ty, const uint32_t* bg) {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPLOADI(L0, kFloatB, 0);       // depth 0
        TTI_SFPLOADI(L4, kFloatB, 0xBF80);  // triangle -1
        load_f32(L1, bg[0]);
        load_f32(L2, bg[1]);
        load_f32(L3, bg[2]);
        for (uint32_t r = 0; r < 32; ++r) {
            store_vec(L0, r);
            store_vec(L1, 32 + r);
            store_vec(L2, 64 + r);
            store_vec(L3, 96 + r);
            store_vec(L4, 128 + r);
        }
        // py of row r = 32 ty + r + 0.5, by exact additions of 1.
        const float y0 = 32.0f * ty + 0.5f;
        const float x0 = 32.0f * tx + 0.5f;
        uint32_t yb, xb;
        __builtin_memcpy(&yb, &y0, 4);
        __builtin_memcpy(&xb, &x0, 4);
        load_f32(L5, yb);
        for (uint32_t r = 0; r < 32; ++r) {
            store_vec(L5, kVecPy + r);
            TTI_SFPMAD(L5, ONE, ONE, L5, 0);
        }
        // px of lane l = 32 tx + l + 0.5 = (2 l) * 0.5 + x0, exact.
        TTI_SFPCAST(LANE, L0, 0);
        TTI_SFPLOADI(L1, kFloatB, 0x3F00);
        load_f32(L2, xb);
        TTI_SFPMAD(L0, L1, L2, L0, 0);
        TTI_SFPNOP;
        store_vec(L0, kVecK + kKPx);
    }

    inline void push(uint32_t instr) {
        instrn_buffer[0] = instr;
    }

    // A float32 bit pattern into an LReg: upper half (lower half zeroed), lower half.
    inline void imm(uint32_t lreg, uint32_t bits) {
        push(TT_OP_SFPLOADI(lreg, kFloatB, bits >> 16));
        push(TT_OP_SFPLOADI(lreg, kLower, bits & 0xFFFF));
    }

    // Copies a 32-word record to local memory, loads grouped ahead of their use.
    inline void fetch(uint32_t* w, const volatile uint32_t* rec) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < kSlotWords; i += 8) {
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                v[j] = rec[i + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                w[i + j] = v[j];
            }
        }
    }

    // Edge k of record w: a px into K[kKA + k] (through LReg R), b, c and T broadcast.
    // The box columns are float16 immediates, masked to the 16-bit field.
#define FILL_TRI_EDGE(k, R)                                    \
    imm(R, w[kRecA + k]);                                      \
    TTI_SFPMAD(R, L5, NZ, R, 0);                               \
    if (k == 0) {                                              \
        push(TT_OP_SFPLOADI(L2, kFloatA, w[kRecX0] & 0xFFFF)); \
        push(TT_OP_SFPLOADI(L4, kFloatA, w[kRecX1] & 0xFFFF)); \
        TTI_SFPLOADI(L3, kFloatB, 0xFF80);                     \
        TTI_SFPGT(0, L2, L5, kSetCC);                          \
        TTI_SFPGT(0, L5, L4, kSetCC);                          \
        TTI_SFPLOADI(L3, kFloatB, 0);                          \
        TTI_SFPENCC(3, 0, 0, 10);                              \
        TTI_SFPMAD(R, ONE, L3, R, 0);                          \
    }                                                          \
    TTI_SFPSTORE(R, kInt32, kAm, kaddr(kKA + k));              \
    imm(L2, w[kRecB + k]);                                     \
    TTI_SFPSTORE(L2, kInt32, kAm, kaddr(kKB + k));             \
    imm(L3, w[kRecC + k]);                                     \
    TTI_SFPSTORE(L3, kInt32, kAm, kaddr(kKC + k))

#define FILL_TRI_COLOUR(c, R)                         \
    imm(R, w[kRecCol + 3 * c]);                       \
    TTI_SFPMAD(R, L5, NZ, R, 0);                      \
    imm(L2, w[kRecCol + 3 * c + 1]);                  \
    TTI_SFPSTORE(L2, kInt32, kAm, kaddr(kKColB + c)); \
    imm(L3, w[kRecCol + 3 * c + 2]);                  \
    TTI_SFPSTORE(L3, kInt32, kAm, kaddr(kKColC + c)); \
    TTI_SFPSTORE(R, kInt32, kAm, kaddr(kKColA + c))

    // The broadcast constants of one triangle record (local copy w) into Dst, its
    // thresholds and number into L2 to L5. Edge 0's a px is -inf outside the box.
    inline void load_triangle(const uint32_t* w) {
        TTI_SFPLOAD(L5, kInt32, kAm, kaddr(kKPx));
        FILL_TRI_EDGE(0, L1);
        FILL_TRI_EDGE(1, L0);
        FILL_TRI_EDGE(2, L1);
        imm(L0, w[kRecW]);
        TTI_SFPMAD(L0, L5, NZ, L0, 0);
        imm(L2, w[kRecW + 1]);
        TTI_SFPSTORE(L2, kInt32, kAm, kaddr(kKW + 1));
        imm(L3, w[kRecW + 2]);
        TTI_SFPSTORE(L3, kInt32, kAm, kaddr(kKW + 2));
        TTI_SFPSTORE(L0, kInt32, kAm, kaddr(kKW));
        FILL_TRI_COLOUR(0, L1);
        FILL_TRI_COLOUR(1, L0);
        FILL_TRI_COLOUR(2, L1);
        for (uint32_t k = 0; k < 3; ++k) {
            push(TT_OP_SFPMOV(0, w[kRecT + k] != 0 ? TNEG : ZERO, L2 + k, 0));
        }
        imm(L5, w[kRecTri]);
    }

    // value = (A + B py) + C into L0, the constants at K vectors ka, kb, kc.
#define FILL_PLANE(ka, kb, kc)               \
    TTI_SFPLOAD(L0, kInt32, kAm, kaddr(kb)); \
    TTI_SFPMAD(L0, PY, NZ, L0, 0);           \
    TTI_SFPLOAD(L1, kInt32, kAm, kaddr(ka)); \
    TTI_SFPMAD(L1, ONE, L0, L0, 0);          \
    TTI_SFPLOAD(L1, kInt32, kAm, kaddr(kc)); \
    TTI_SFPMAD(L0, ONE, L1, L0, 0)

#define FILL_EDGE(k)                       \
    FILL_PLANE(kKA + k, kKB + k, kKC + k); \
    TTI_SFPGT(0, L2 + k, L0, kSetCC)

    // A row's edge tests and 1/z plane: 27 instructions, recorded once into the replay
    // buffer (record) or played (not record).
    constexpr uint32_t kReplayEdges = 27;

    inline void edges(bool record) {
        if (record) {
            TTI_REPLAY(0, kReplayEdges, 0, 1);
            FILL_EDGE(0);
            FILL_EDGE(1);
            FILL_EDGE(2);
            FILL_PLANE(kKW, kKW + 1, kKW + 2);
        } else {
            TTI_REPLAY(0, kReplayEdges, 0, 0);
        }
    }

    // Row r (masked to its field) against the triangle in the K vectors: the inside
    // lanes nearer than the depth take its depth, number and colour.
    inline void row(uint32_t r) {
        r &= 31;
        TT_SFPLOAD(PY, kInt32, kAm, addr(kVecPy + r));
        edges(false);
        TT_SFPLOAD(L1, kInt32, kAm, addr(r));
        TTI_SFPGT(0, L1, L0, kSetCC);
        TT_SFPSTORE(L0, kInt32, kAm, addr(r));
        TT_SFPSTORE(L5, kInt32, kAm, addr(128 + r));
        FILL_PLANE(kKColA, kKColB, kKColC);
        TT_SFPSTORE(L0, kInt32, kAm, addr(32 + r));
        FILL_PLANE(kKColA + 1, kKColB + 1, kKColC + 1);
        TT_SFPSTORE(L0, kInt32, kAm, addr(64 + r));
        FILL_PLANE(kKColA + 2, kKColB + 2, kKColC + 2);
        TT_SFPSTORE(L0, kInt32, kAm, addr(96 + r));
        TTI_SFPENCC(3, 0, 0, 10);
    }

    // ARITH: 16 vectors of inputs in Dst vectors [16 k, 16 k + 16) for input k (a, b,
    // c, px, py, T, d); outputs j from vector 112 + 16 j.
    inline void arith_chunk() {
        for (uint32_t i = 0; i < 16; ++i) {
            TTI_SFPENCC(3, 0, 0, 10);
            load_vec(L0, i);
            load_vec(L1, 48 + i);
            TTI_SFPMAD(L0, L1, NZ, L2, 0);
            load_vec(L0, 16 + i);
            load_vec(L1, 64 + i);
            TTI_SFPMAD(L0, L1, NZ, L3, 0);
            TTI_SFPNOP;
            store_vec(L2, 112 + i);
            store_vec(L3, 128 + i);
            TTI_SFPMAD(L2, ONE, L3, L2, 0);
            load_vec(L0, 32 + i);
            store_vec(L2, 144 + i);
            TTI_SFPMAD(L2, ONE, L0, L2, 0);
            TTI_SFPNOP;
            store_vec(L2, 160 + i);
            TTI_SFPLOADI(L4, kUShort, 0);
            TTI_SFPLOADI(L5, kUShort, 0);
            load_vec(L0, 80 + i);
            TTI_SFPGT(0, L0, L2, kSetCC);
            TTI_SFPLOADI(L4, kUShort, 1);
            TTI_SFPENCC(3, 0, 0, 10);
            load_vec(L0, 96 + i);
            TTI_SFPGT(0, L0, L2, kSetCC);
            TTI_SFPLOADI(L5, kUShort, 1);
            TTI_SFPENCC(3, 0, 0, 10);
            store_vec(L4, 176 + i);
            store_vec(L5, 192 + i);
        }
    }

}  // namespace fill
