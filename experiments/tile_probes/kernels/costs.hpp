// Probes 8 to 10 and 13, run by the pack thread (TRISC2): the cost of an instruction
// push, replay and MOP loops, load-macro chains, SFPCONFIG combine modes, SFPI code.
#pragma once

#include <stdint.h>

#include "ckernel_template.h"
#include "probe.hpp"
#include "sfpi.h"
#include "vec.hpp"

namespace costs {

    using namespace vec;

    // One timed span of the pushes in f, between two tensix_sync calls.
    template <typename F>
    inline uint32_t span(F f) {
        tensix_sync();
        const uint32_t a = wall();
        f();
        tensix_sync();
        return wall() - a;
    }

    // A value the compiler cannot see through.
    inline uint32_t opaque(uint32_t v) {
        asm volatile("" : "+r"(v));
        return v;
    }

    // Stores L0..L3 (or only L4 when one) to Dst slot s: addresses 16 s + 4 r.
    inline void store_slot(uint32_t s, bool one) {
        if (one) {
            TT_SFPSTORE(L4, kInt32, kAm, 16 * s);
        } else {
            TT_SFPSTORE(L0, kInt32, kAm, 16 * s);
            TT_SFPSTORE(L1, kInt32, kAm, 16 * s + 4);
            TT_SFPSTORE(L2, kInt32, kAm, 16 * s + 8);
            TT_SFPSTORE(L3, kInt32, kAm, 16 * s + 12);
        }
        tensix_sync();
    }

    // ---- Probe 8: push forms (row 1) and replay / MOP loops (row 2) ----------------

#define CP_X4(m) \
    m;           \
    m;           \
    m;           \
    m
#define CP_X16(m) CP_X4(CP_X4(m))
#define CP_X64(m) CP_X4(CP_X16(m))
#define CP_LOADI TTI_SFPLOADI(L4, 2, 0x1234)
#define CP_STORE instrn_buffer[0] = w
#define CP_MAD4 \
    TP_MAD(L0); \
    TP_MAD(L1); \
    TP_MAD(L2); \
    TP_MAD(L3)

    constexpr uint32_t kLoadi = TT_OP_SFPLOADI(L4, 2, 0);  // USHORT immediate into L4
    constexpr uint32_t kPushForms = 7, kLoopForms = 6;

    // Row 1 form f, r loop iterations of 16 pushes (64 for forms 2 and 4) of SFPLOADI.
    // Returns the immediate L4 holds afterwards.
    [[gnu::noinline]] uint32_t push_form(
        uint32_t f, uint32_t r, volatile uint32_t* table, Times& t
    ) {
        uint32_t last = 0;
        if (f == 0) {
            t.add(span([] {}));
        } else if (f == 1) {
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    CP_X16(CP_LOADI);
                }
            }));
            last = 0x1234;
        } else if (f == 2) {
            t.add(span([&] {
                for (uint32_t i = 0; i < r / 4; ++i) {
                    CP_X64(CP_LOADI);
                }
            }));
            last = 0x1234;
        } else if (f == 3 || f == 4) {
            const uint32_t w = opaque(kLoadi | 0x2345);
            const uint32_t n = f == 3 ? r : r / 4;
            t.add(span([&] {
                for (uint32_t i = 0; i < n; ++i) {
                    if (f == 3) {
                        CP_X16(CP_STORE);
                    } else {
                        CP_X64(CP_STORE);
                    }
                }
            }));
            last = 0x2345;
        } else if (f == 5) {
            // Immediate built from a run-time value, masked to its 16-bit field.
            uint32_t x = opaque(7);
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    CP_X16(x += 0x9E37; instrn_buffer[0] = kLoadi + (x & 0xFFFF));
                }
            }));
            last = x & 0xFFFF;
        } else {
            // Words encoded ahead of time in an L1 table of 64.
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    volatile uint32_t* p = table + 16 * (i & 3);
                    CP_X16(instrn_buffer[0] = *p++);
                }
            }));
            last = (((r - 1) & 3) * 16 + 15) * 77 & 0xFFFF;
        }
        return last;
    }

    // Row 2 form f over the body of 16 SFPMAD (4 registers): pushed r times, one replay
    // of 16 r times, replays of 4 (4 r), MOP over replays of 16 / of 4, MOP of 2 MADs.
    [[gnu::noinline]] void loop_form(uint32_t f, uint32_t r, Times& t) {
        const uint32_t rep16 = TT_OP_REPLAY(0, 16, 0, 0);
        const uint32_t rep4 = TT_OP_REPLAY(16, 4, 0, 0);
        if (f == 0) {
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    CP_X4(CP_MAD4);
                }
            }));
        } else if (f == 1) {
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    instrn_buffer[0] = rep16;
                }
            }));
        } else if (f == 2) {
            t.add(span([&] {
                for (uint32_t i = 0; i < r; ++i) {
                    CP_X4(instrn_buffer[0] = rep4);
                }
            }));
        } else if (f == 3 || f == 4) {
            // r = outer x 32 replays of 16, or 4 r = outer x 64 replays of 4.
            ckernel_template mop(f == 3 ? r / 32 : r / 16, f == 3 ? 32 : 64, f == 3 ? rep16 : rep4);
            mop.program();
            t.add(span([] { ckernel_template::run(); }));
        } else {
            // 16 r instructions = outer x inner x 2 (LoopOp and LoopOp1 alternate).
            ckernel_template mop(
                r / 8, 64, TT_OP_SFPMAD(L0, ONE, ONE, L0, 0), TT_OP_SFPMAD(L1, ONE, ONE, L1, 0)
            );
            mop.program();
            t.add(span([] { ckernel_template::run(); }));
        }
    }

    // Timing: K spans per push form, then K per loop form. Data: L4 of each push form
    // at slot f, L0..L3 of each loop form at slot 8 + f; expected immediates at word 2048.
    inline void push(volatile uint32_t* out, volatile uint32_t* scratch, uint32_t r, uint32_t k_runs, Times& t) {
        for (uint32_t j = 0; j < 64; ++j) {
            scratch[j] = kLoadi | ((j * 77) & 0xFFFF);
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t f = 0; f < kPushForms; ++f) {
            uint32_t last = 0;
            for (uint32_t k = 0; k < k_runs; ++k) {
                TTI_SFPLOADI(L4, 2, 0);
                last = push_form(f, r, scratch, t);
            }
            out[probe::kData + 2048 + f] = last;
            store_slot(f, true);
        }
        TTI_REPLAY(0, 16, 0, 1);
        CP_X4(CP_MAD4);
        TTI_REPLAY(16, 4, 0, 1);
        CP_MAD4;
        for (uint32_t f = 0; f < kLoopForms; ++f) {
            for (uint32_t k = 0; k < k_runs; ++k) {
                pred_registers();
                loop_form(f, r, t);
            }
            store_slot(8 + f, false);
        }
        for (uint32_t s = 0; s < 8 + kLoopForms; ++s) {
            for (uint32_t q = 0; q < 4; ++q) {
                read_lanes(16 * s + 4 * q, out + probe::kData + 128 * s + 32 * q);
            }
        }
    }

    // ---- Probe 9: load-macro chain (row 5) -------------------------------------------

    constexpr uint32_t kRows = 64;  // vectors at Dst addresses 0, 2, .., 126

    // Dst addresses 2 k for k < 64 from the input words in[32 k + l].
    inline void macro_input(const volatile uint32_t* in) {
        for (uint32_t k = 0; k < kRows; ++k) {
            write_lanes(2 * k, in + 32 * k);
        }
    }

    // Form f, p passes over the 64 vectors (x += 1 each): 0 one instruction at a time
    // (load, add, store a vector), 1 four vectors interleaved, 2 one load macro a vector.
    [[gnu::noinline]] void macro_form(uint32_t f, uint32_t p, Times& t) {
        if (f == 0) {
            TTI_REPLAY(0, 3, 0, 1);
            TTI_SFPLOAD(L0, kInt32, kAm, 0);
            TP_MAD(L0);
            TTI_SFPSTORE(L0, kInt32, ADDR_MOD_6, 0);
        } else if (f == 1) {
            TTI_REPLAY(0, 12, 0, 1);
            TTI_SFPLOAD(L0, kInt32, kAm, 0);
            TTI_SFPLOAD(L1, kInt32, kAm, 2);
            TTI_SFPLOAD(L2, kInt32, kAm, 4);
            TTI_SFPLOAD(L3, kInt32, kAm, 6);
            CP_MAD4;
            TTI_SFPSTORE(L0, kInt32, kAm, 0);
            TTI_SFPSTORE(L1, kInt32, kAm, 2);
            TTI_SFPSTORE(L2, kInt32, kAm, 4);
            TTI_SFPSTORE(L3, kInt32, ADDR_MOD_5, 6);
        } else {
            TTI_REPLAY(0, 4, 0, 1);
            TTI_SFPLOADMACRO(0, kInt32, ADDR_MOD_6, 0);
            TTI_SFPLOADMACRO(1, kInt32, ADDR_MOD_6, 0);
            TTI_SFPLOADMACRO(2, kInt32, ADDR_MOD_6, 0);
            TTI_SFPLOADMACRO(3, kInt32, ADDR_MOD_6, 0);
        }
        const uint32_t len = f == 0 ? 3 : f == 1 ? 12 : 4;
        const uint32_t per = f == 0 ? kRows : kRows / 4;
        const uint32_t rep = TT_OP_REPLAY(0, len, 0, 0);
        t.add(span([&] {
            for (uint32_t q = 0; q < p; ++q) {
                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                for (uint32_t i = 0; i < per; ++i) {
                    instrn_buffer[0] = rep;
                }
            }
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
            TTI_SFPNOP;
        }));
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

    // Macro 0: load [x] into the register named by the instruction, MAD slot (delay 0)
    // x = 1 * 1 + x from template 0, store slot (delay 2) x back to the macro's address.
    inline void macro_setup() {
        TTI_SFPCONFIG(0, 0xF, 1);  // LaneConfig 0: back-door template load on
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPMAD(ONE, ONE, 0, 12, 0);  // template 0 (back door, VD 12)
        constexpr uint32_t mad_bits = (0 << 3) | 4;
        constexpr uint32_t store_bits = (2 << 3) | 3;
        TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_LOWER, mad_bits << 8);
        TTI_SFPLOADI(L0, sfpi::SFPLOADI_MOD0_UPPER, store_bits << 8);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x010, 8, 1);  // Misc: macro 0 stores in the load's format
        TTI_SFPNOP;
        TTI_SFPNOP;
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_5);
        tensix_sync();
    }

    // Sequence 0 and Misc back to zero (no macro left armed).
    inline void macro_clear() {
        TTI_SFPLOADI(L0, 2, 0);
        TTI_SFPCONFIG(0, 4, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        tensix_sync();
    }

    // Timing: K spans per form. Data: per form the count of wrong lanes and the first
    // (word 3 f, 3 f + 1), then the 64 vectors after the last form at word 64.
    inline void macro(const volatile uint32_t* in, volatile uint32_t* out, uint32_t p, uint32_t k_runs, Times& t) {
        volatile uint32_t* d = out + probe::kData;
        TTI_SFPMOV(0, 15, L7, 8);  // LaneConfig as found, restored at the end
        TTI_SFPNOP;
        macro_setup();
        macro_input(in);
        uint32_t n = 0;
        for (uint32_t f = 0; f < 3; ++f) {
            for (uint32_t k = 0; k < k_runs; ++k) {
                macro_form(f, p, t);
            }
            n += k_runs * p;
            tensix_sync();
            volatile uint32_t* w = win();
            uint32_t bad = 0, first = ~0u;
            for (uint32_t k = 0; k < kRows; ++k) {
                for (uint32_t l = 0; l < 32; ++l) {
                    float x, want;
                    const uint32_t bits = in[32 * k + l], got = w[lane_word(2 * k, l)];
                    __builtin_memcpy(&x, &bits, 4);
                    want = x + static_cast<float>(n);
                    uint32_t want_bits;
                    __builtin_memcpy(&want_bits, &want, 4);
                    if (got != want_bits && bad++ == 0) {
                        first = 32 * k + l;
                    }
                }
            }
            d[3 * f] = bad;
            d[3 * f + 1] = first;
            d[3 * f + 2] = n;
        }
        for (uint32_t k = 0; k < kRows; ++k) {
            read_lanes(2 * k, d + 64 + 32 * k);
        }
        macro_clear();
        TTI_SFPMOV(0, L7, L0, 0);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 15, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        tensix_sync();
    }

    // ---- Probe 10: SFPCONFIG combine modes (row 8) -----------------------------------

    // Misc (special 8) or LaneConfig (special 15) into L0, stored to Dst slot s.
    inline void read_special(uint32_t which, uint32_t s) {
        TTI_SFPNOP;
        TTI_SFPNOP;
        if (which == 8) {
            TTI_SFPMOV(0, 8, L0, 8);
        } else {
            TTI_SFPMOV(0, 15, L0, 8);
        }
        TTI_SFPNOP;
        TT_SFPSTORE(L0, kInt32, kAm, 16 * s);
        tensix_sync();
    }

    // Slots: 0 Misc and 1 LaneConfig as found; Misc after replace 0x0F0, mode 2, 4, 6
    // with 0x0CC (slots 2..5); LaneConfig after mode 2 (0x0001 with 0x0100) and mode 4
    // (0x0101 with 0x0001) (slots 6, 7); both after the restore (slots 8, 9).
    inline void sfpcfg(volatile uint32_t* out) {
        TTI_SFPMOV(0, 8, L6, 8);
        TTI_SFPMOV(0, 15, L7, 8);
        TTI_SFPNOP;
        TTI_SFPSTORE(L6, kInt32, kAm, 0);
        TTI_SFPSTORE(L7, kInt32, kAm, 16);
        tensix_sync();
        TTI_SFPCONFIG(0x0F0, 8, 1);
        read_special(8, 2);
        TTI_SFPCONFIG(0x0CC, 8, 1 | 2);
        read_special(8, 3);
        TTI_SFPCONFIG(0x0F0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x0CC, 8, 1 | 4);
        read_special(8, 4);
        TTI_SFPCONFIG(0x0F0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x0CC, 8, 1 | 6);
        read_special(8, 5);
        TTI_SFPCONFIG(0x0001, 15, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x0100, 15, 1 | 2);
        read_special(15, 6);
        TTI_SFPCONFIG(0x0101, 15, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0x0001, 15, 1 | 4);
        read_special(15, 7);
        // Restore both from the saved registers (replace from LReg 0).
        TTI_SFPMOV(0, L7, L0, 0);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 15, 0);
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPMOV(0, L6, L0, 0);
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 8, 0);
        read_special(8, 8);
        read_special(15, 9);
        for (uint32_t s = 0; s < 10; ++s) {
            read_lanes(16 * s, out + probe::kData + 32 * s);
        }
    }

    // ---- Probe 13: what the SFPI compiler emits (row 6) ------------------------------

    // Inputs a, b, c at dst_reg[0..2]; results at dst_reg[3..6].
    [[gnu::noinline]] void sfpi_mad() {
        sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[1], c = sfpi::dst_reg[2];
        sfpi::dst_reg[3] = a * b + c;
    }

    [[gnu::noinline]] void sfpi_sub() {
        sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[1];
        sfpi::dst_reg[4] = a - b;
    }

    [[gnu::noinline]] void sfpi_const() {
        sfpi::vFloat a = sfpi::dst_reg[0];
        sfpi::dst_reg[5] = a * 1.5f + 0.25f;
    }

    [[gnu::noinline]] void sfpi_if() {
        sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[1], c = sfpi::dst_reg[2];
        sfpi::vFloat r = a;
        v_if (a < b) {
            r = c;
        }
        v_endif;
        sfpi::dst_reg[6] = r;
    }

    // Input words in[32 i + l] for i < 3 into dst_reg[i]; dst_reg[0..6] back at data.
    inline void sfpi_run(const volatile uint32_t* in, volatile uint32_t* out) {
        for (uint32_t i = 0; i < 3; ++i) {
            write_lanes(2 * i, in + 32 * i);
        }
        sfpi_mad();
        sfpi_sub();
        sfpi_const();
        sfpi_if();
        tensix_sync();
        for (uint32_t i = 0; i < 7; ++i) {
            read_lanes(2 * i, out + probe::kData + 32 * i);
        }
    }

}  // namespace costs
