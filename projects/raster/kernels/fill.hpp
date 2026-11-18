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
    constexpr uint32_t kAddressMode = ADDR_MOD_7;
    constexpr uint32_t kFloatB = 0, kFloatA = 1, kUShort = 2, kLower = 10;
    constexpr uint32_t kSetCC = 1;  // SFPGT: lanes stay enabled where VD > VC
    constexpr uint32_t kWallClock = 0xFFB121F0;
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    // Dst vectors: plane p row r at 32 p + r (depth, red, green, blue, triangle), py of
    // row r at 160 + r, the triangle's broadcast constants from 192.
    constexpr uint32_t kVecPy = 160;
    constexpr uint32_t kVecConstants = 192;
    constexpr uint32_t kConstA = 0;
    constexpr uint32_t kConstB = 3;
    constexpr uint32_t kConstC = 6;
    constexpr uint32_t kConstW = 9;
    constexpr uint32_t kConstColourA = 12;
    constexpr uint32_t kConstColourB = 15;
    constexpr uint32_t kConstColourC = 18;
    constexpr uint32_t kConstPx = 21;

    // Dst address of `vector`: an aligned group of four rows, even or odd columns.
    constexpr uint32_t dst_address(uint32_t vector) {
        return 4 * (vector >> 1) + 2 * (vector & 1);
    }

    constexpr uint32_t constant_address(uint32_t index) {
        return dst_address(kVecConstants + index);
    }

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    // `words` words (a multiple of 8), loads grouped ahead of the stores that use them.
    inline void copy_words(
        volatile uint32_t* destination, const volatile uint32_t* source, uint32_t words
    ) {
        for (uint32_t i = 0; i < words; i += 8) {
            uint32_t loaded[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                loaded[j] = source[i + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                destination[i + j] = loaded[j];
            }
        }
    }

#if defined(__riscv_vector)
    // The same copy on the pack core's vector extension, 32 words per instruction.
    inline void copy_words_rvv(
        uint32_t* destination, const uint32_t* source, uint32_t words
    ) {
        const size_t vector_length = __riscv_vsetvl_e32m8(32);
        for (uint32_t i = 0; i < words; i += 32) {
            __riscv_vse32_v_u32m8(
                destination + i,
                __riscv_vle32_v_u32m8(source + i, vector_length),
                vector_length
            );
        }
        asm volatile("" ::: "memory");
    }
#endif

    // Waits until earlier window stores are done (a load of the last word, used).
    inline void order(uint32_t word) {
        uint32_t last_word = win()[word];
        asm volatile("and x0, x0, %0" : : "r"(last_word) : "memory");
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

    inline void load_vec(uint32_t lreg, uint32_t vector) {
        TT_SFPLOAD(lreg, kInt32, kAddressMode, dst_address(vector));
    }

    inline void store_vec(uint32_t lreg, uint32_t vector) {
        TT_SFPSTORE(lreg, kInt32, kAddressMode, dst_address(vector));
    }

    // Cycles of the pack thread, by phase (wall clock).
    struct ProfileCounters {
        uint32_t wait_in = 0;         // waiting for list pages
        uint32_t init = 0;            // planes, px and py of a tile
        uint32_t triangle_setup = 0;  // a triangle's constants
        uint32_t rows = 0;            // the row bodies
        uint32_t drain = 0;           // vector unit finishing before the read-out
        uint32_t out = 0;             // planes from Dst to the output pages
        uint32_t wait_out = 0;        // waiting for free output pages
        uint32_t tiles = 0;
        uint32_t triangles = 0;
        uint32_t row_count = 0;
        uint32_t total = 0;
#ifdef RASTER_SPREAD
        uint32_t landing_wait = 0;  // waiting for a spread tile in Dst
#endif
    };

    // Planes of tile (tile_x, tile_y) to their empty values; py of its rows and px of
    // its columns into Dst.
    inline void init_tile(
        uint32_t tile_x, uint32_t tile_y, const uint32_t* background
    ) {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPLOADI(L0, kFloatB, 0);       // depth 0
        TTI_SFPLOADI(L4, kFloatB, 0xBF80);  // triangle -1
        load_f32(L1, background[0]);
        load_f32(L2, background[1]);
        load_f32(L3, background[2]);
        for (uint32_t row = 0; row < 32; ++row) {
            store_vec(L0, row);
            store_vec(L1, 32 + row);
            store_vec(L2, 64 + row);
            store_vec(L3, 96 + row);
            store_vec(L4, 128 + row);
        }
        // py of a row = 32 tile_y + row + 0.5, by exact additions of 1.
        const float origin_y = 32.0f * tile_y + 0.5f;
        const float origin_x = 32.0f * tile_x + 0.5f;
        uint32_t origin_y_bits, origin_x_bits;
        __builtin_memcpy(&origin_y_bits, &origin_y, 4);
        __builtin_memcpy(&origin_x_bits, &origin_x, 4);
        load_f32(L5, origin_y_bits);
        for (uint32_t row = 0; row < 32; ++row) {
            store_vec(L5, kVecPy + row);
            TTI_SFPMAD(L5, ONE, ONE, L5, 0);
        }
        // px of lane l = 32 tile_x + l + 0.5 = (2 l) * 0.5 + origin_x, exact.
        TTI_SFPCAST(LANE, L0, 0);
        TTI_SFPLOADI(L1, kFloatB, 0x3F00);
        load_f32(L2, origin_x_bits);
        TTI_SFPMAD(L0, L1, L2, L0, 0);
        TTI_SFPNOP;
        store_vec(L0, kVecConstants + kConstPx);
    }

    inline void push(uint32_t instruction) {
        instrn_buffer[0] = instruction;
    }

    // A float32 bit pattern into an LReg: upper half (lower half zeroed), lower half.
    inline void immediate(uint32_t lreg, uint32_t bits) {
        push(TT_OP_SFPLOADI(lreg, kFloatB, bits >> 16));
        push(TT_OP_SFPLOADI(lreg, kLower, bits & 0xFFFF));
    }

    // Copies a 32-word record to local memory, loads grouped ahead of their use.
    inline void fetch(uint32_t* words, const volatile uint32_t* record) {
#pragma GCC unroll 4
        for (uint32_t i = 0; i < kSlotWords; i += 8) {
            uint32_t loaded[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                loaded[j] = record[i + j];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                words[i + j] = loaded[j];
            }
        }
    }

    // Edge `edge` of `record_words`: a px into constants[kConstA + edge] (through LReg
    // `lreg`), b, c and T broadcast. The box columns are masked float16 immediates.
#define FILL_TRI_EDGE(edge, lreg)                                               \
    immediate(lreg, record_words[kRecA + edge]);                                \
    TTI_SFPMAD(lreg, L5, NZ, lreg, 0);                                          \
    if (edge == 0) {                                                            \
        push(TT_OP_SFPLOADI(L2, kFloatA, record_words[kRecX0] & 0xFFFF));       \
        push(TT_OP_SFPLOADI(L4, kFloatA, record_words[kRecX1] & 0xFFFF));       \
        TTI_SFPLOADI(L3, kFloatB, 0xFF80);                                      \
        TTI_SFPGT(0, L2, L5, kSetCC);                                           \
        TTI_SFPGT(0, L5, L4, kSetCC);                                           \
        TTI_SFPLOADI(L3, kFloatB, 0);                                           \
        TTI_SFPENCC(3, 0, 0, 10);                                               \
        TTI_SFPMAD(lreg, ONE, L3, lreg, 0);                                     \
    }                                                                           \
    TTI_SFPSTORE(lreg, kInt32, kAddressMode, constant_address(kConstA + edge)); \
    immediate(L2, record_words[kRecB + edge]);                                  \
    TTI_SFPSTORE(L2, kInt32, kAddressMode, constant_address(kConstB + edge));   \
    immediate(L3, record_words[kRecC + edge]);                                  \
    TTI_SFPSTORE(L3, kInt32, kAddressMode, constant_address(kConstC + edge))

#define FILL_TRI_COLOUR(colour, lreg)                                                 \
    immediate(lreg, record_words[kRecCol + 3 * colour]);                              \
    TTI_SFPMAD(lreg, L5, NZ, lreg, 0);                                                \
    immediate(L2, record_words[kRecCol + 3 * colour + 1]);                            \
    TTI_SFPSTORE(L2, kInt32, kAddressMode, constant_address(kConstColourB + colour)); \
    immediate(L3, record_words[kRecCol + 3 * colour + 2]);                            \
    TTI_SFPSTORE(L3, kInt32, kAddressMode, constant_address(kConstColourC + colour)); \
    TTI_SFPSTORE(lreg, kInt32, kAddressMode, constant_address(kConstColourA + colour))

    // The broadcast constants of one triangle record (local copy `record_words`) into
    // Dst, its thresholds and number into L2 to L5. Edge 0's a px is -inf off the box.
    inline void load_triangle(const uint32_t* record_words) {
        TTI_SFPLOAD(L5, kInt32, kAddressMode, constant_address(kConstPx));
        FILL_TRI_EDGE(0, L1);
        FILL_TRI_EDGE(1, L0);
        FILL_TRI_EDGE(2, L1);
        immediate(L0, record_words[kRecW]);
        TTI_SFPMAD(L0, L5, NZ, L0, 0);
        immediate(L2, record_words[kRecW + 1]);
        TTI_SFPSTORE(L2, kInt32, kAddressMode, constant_address(kConstW + 1));
        immediate(L3, record_words[kRecW + 2]);
        TTI_SFPSTORE(L3, kInt32, kAddressMode, constant_address(kConstW + 2));
        TTI_SFPSTORE(L0, kInt32, kAddressMode, constant_address(kConstW));
        FILL_TRI_COLOUR(0, L1);
        FILL_TRI_COLOUR(1, L0);
        FILL_TRI_COLOUR(2, L1);
        for (uint32_t edge = 0; edge < 3; ++edge) {
            push(TT_OP_SFPMOV(
                0, record_words[kRecT + edge] != 0 ? TNEG : ZERO, L2 + edge, 0
            ));
        }
        immediate(L5, record_words[kRecTri]);
    }

#ifdef RASTER_SPREAD
    // The spread tile's 21 vectors (the constants' order) in Dst tile 7, its rows.
    constexpr uint32_t kVecLanding = 224;

    constexpr uint32_t landing_address(uint32_t index) {
        return dst_address(kVecLanding + index);
    }

    static_assert(landing_address(0) == kSpreadDstRow);

#define FILL_SPREAD_PRODUCT(index, lreg)                              \
    TTI_SFPLOAD(lreg, kInt32, kAddressMode, constant_address(index)); \
    TTI_SFPMAD(lreg, L5, NZ, lreg, 0);                                \
    TTI_SFPSTORE(lreg, kInt32, kAddressMode, constant_address(index))

    // Slot `slot` into L0 to L3; SFPTRANSP puts its row k (vector 4 slot + k) in every
    // lane of L_k, stored but for padding; a second SFPTRANSP gives L4 to L7 back.
    template <uint32_t slot>
    inline void spread_slot() {
        TTI_SFPLOAD(L0, kInt32, kAddressMode, landing_address(slot));
        TTI_SFPLOAD(L1, kInt32, kAddressMode, landing_address(slot));
        TTI_SFPLOAD(L2, kInt32, kAddressMode, landing_address(slot));
        TTI_SFPLOAD(L3, kInt32, kAddressMode, landing_address(slot));
        TTI_SFPTRANSP(0, 0, 0, 0);
        TTI_SFPSTORE(L0, kInt32, kAddressMode, constant_address(4 * slot));
        if constexpr (4 * slot + 1 < kSpreadVectors) {
            TTI_SFPSTORE(L1, kInt32, kAddressMode, constant_address(4 * slot + 1));
            TTI_SFPSTORE(L2, kInt32, kAddressMode, constant_address(4 * slot + 2));
            TTI_SFPSTORE(L3, kInt32, kAddressMode, constant_address(4 * slot + 3));
        }
        TTI_SFPTRANSP(0, 0, 0, 0);
    }

    // load_triangle from the landed tile: each of the 21 numbers to every lane, the
    // tile given back (semaphore `free`), then a px formed into the a vectors.
    inline void load_spread(const uint32_t* record_words, uint32_t free) {
        static_assert(kSpreadSlotVectors == 24 && kSpreadVectors == 21);
        spread_slot<0>();
        spread_slot<1>();
        spread_slot<2>();
        spread_slot<3>();
        spread_slot<4>();
        spread_slot<5>();
        t6_semaphore_post<p_stall::WAIT_SFPU>(free);
        TTI_SFPLOAD(L5, kInt32, kAddressMode, constant_address(kConstPx));
        TTI_SFPLOAD(L1, kInt32, kAddressMode, constant_address(kConstA));
        TTI_SFPMAD(L1, L5, NZ, L1, 0);
        push(TT_OP_SFPLOADI(L2, kFloatA, record_words[kRecX0] & 0xFFFF));
        push(TT_OP_SFPLOADI(L4, kFloatA, record_words[kRecX1] & 0xFFFF));
        TTI_SFPLOADI(L3, kFloatB, 0xFF80);
        TTI_SFPGT(0, L2, L5, kSetCC);
        TTI_SFPGT(0, L5, L4, kSetCC);
        TTI_SFPLOADI(L3, kFloatB, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMAD(L1, ONE, L3, L1, 0);
        TTI_SFPSTORE(L1, kInt32, kAddressMode, constant_address(kConstA));
        FILL_SPREAD_PRODUCT(kConstA + 1, L0);
        FILL_SPREAD_PRODUCT(kConstA + 2, L1);
        FILL_SPREAD_PRODUCT(kConstW, L0);
        FILL_SPREAD_PRODUCT(kConstColourA, L1);
        FILL_SPREAD_PRODUCT(kConstColourA + 1, L0);
        FILL_SPREAD_PRODUCT(kConstColourA + 2, L1);
        for (uint32_t edge = 0; edge < 3; ++edge) {
            push(TT_OP_SFPMOV(
                0, record_words[kRecT + edge] != 0 ? TNEG : ZERO, L2 + edge, 0
            ));
        }
        immediate(L5, record_words[kRecTri]);
    }
#endif

    // value = (A + B py) + C into L0, the constants at the vectors `constant_a`,
    // `constant_b`, `constant_c`.
#define FILL_PLANE(constant_a, constant_b, constant_c)                   \
    TTI_SFPLOAD(L0, kInt32, kAddressMode, constant_address(constant_b)); \
    TTI_SFPMAD(L0, PY, NZ, L0, 0);                                       \
    TTI_SFPLOAD(L1, kInt32, kAddressMode, constant_address(constant_a)); \
    TTI_SFPMAD(L1, ONE, L0, L0, 0);                                      \
    TTI_SFPLOAD(L1, kInt32, kAddressMode, constant_address(constant_c)); \
    TTI_SFPMAD(L0, ONE, L1, L0, 0)

#define FILL_EDGE(edge)                                         \
    FILL_PLANE(kConstA + edge, kConstB + edge, kConstC + edge); \
    TTI_SFPGT(0, L2 + edge, L0, kSetCC)

    // A row's edge tests and 1/z plane: 27 instructions, recorded once into the replay
    // buffer (record) or played (not record).
    constexpr uint32_t kReplayEdges = 27;

    inline void edges(bool record) {
        if (record) {
            TTI_REPLAY(0, kReplayEdges, 0, 1);
            FILL_EDGE(0);
            FILL_EDGE(1);
            FILL_EDGE(2);
            FILL_PLANE(kConstW, kConstW + 1, kConstW + 2);
        } else {
            TTI_REPLAY(0, kReplayEdges, 0, 0);
        }
    }

    // Row `pixel_row` (masked to its field) against the triangle in the constant
    // vectors: the inside lanes nearer than the depth take its depth, number, colour.
    inline void row(uint32_t pixel_row) {
        pixel_row &= 31;
        TT_SFPLOAD(PY, kInt32, kAddressMode, dst_address(kVecPy + pixel_row));
        edges(false);
        TT_SFPLOAD(L1, kInt32, kAddressMode, dst_address(pixel_row));
        TTI_SFPGT(0, L1, L0, kSetCC);
        TT_SFPSTORE(L0, kInt32, kAddressMode, dst_address(pixel_row));
        TT_SFPSTORE(L5, kInt32, kAddressMode, dst_address(128 + pixel_row));
        FILL_PLANE(kConstColourA, kConstColourB, kConstColourC);
        TT_SFPSTORE(L0, kInt32, kAddressMode, dst_address(32 + pixel_row));
        FILL_PLANE(kConstColourA + 1, kConstColourB + 1, kConstColourC + 1);
        TT_SFPSTORE(L0, kInt32, kAddressMode, dst_address(64 + pixel_row));
        FILL_PLANE(kConstColourA + 2, kConstColourB + 2, kConstColourC + 2);
        TT_SFPSTORE(L0, kInt32, kAddressMode, dst_address(96 + pixel_row));
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
