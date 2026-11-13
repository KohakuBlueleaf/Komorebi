// Vector-unit probes 0 to 4, submitted by the pack thread (TRISC2) with raw Tensix
// instructions; Dst (32-bit, INT32 mode) is read and written through the window.
#pragma once

#include <stdint.h>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_dest.h"
#include "ckernel_ops.h"
#include "probe.hpp"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace vec {

    using namespace ckernel;

    constexpr uint32_t L0 = 0, L1 = 1, L2 = 2, L3 = 3, L4 = 4, L5 = 5, L6 = 6, L7 = 7;
    constexpr uint32_t ONE = 10, LANE = 15;      // LReg 10 = 1.0, LReg 15 = 2 x lane
    constexpr uint32_t kInt32 = 4;               // SFPLOAD / SFPSTORE MOD0_FMT_INT32
    constexpr uint32_t kAm = ADDR_MOD_7;         // set to no step by setup()
    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L
    constexpr uint32_t kDstWindow = 0xFFBD8000;

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline uint32_t mcycle() {
        uint32_t c;
        asm volatile("csrr %0, mcycle" : "=r"(c));
        return c;
    }

    // tt_cfg_qstatus (0xBC0) without a fence: bit 11 = some vector lane enabled.
    inline uint32_t qstatus_raw() {
        uint32_t v;
        asm volatile("csrr %0, 0xBC0" : "=r"(v));
        return v;
    }

    // tt_cfg_bstatus (0xBC1) without a fence: bit 9 = this thread's SFPU busy.
    inline uint32_t bstatus_raw() {
        uint32_t v;
        asm volatile("csrr %0, 0xBC1" : "=r"(v));
        return v;
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    constexpr uint32_t lane_word(uint32_t a, uint32_t l) {
        return 16 * ((a & ~3u) + (l >> 3)) + 2 * (l & 7) + ((a >> 1) & 1);
    }

    // Waits until earlier window stores are done: a load of the last word written,
    // consumed by an ALU instruction (same memory region, processed in order).
    inline void order(uint32_t word) {
        uint32_t v = win()[word];
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
    }

    // Timing words: out[kTimes + i], count in out[kNumTimes].
    struct Times {
        volatile uint32_t* out;
        uint32_t n = 0;

        void add(uint32_t v) {
            if (n < probe::kMaxTimes) {
                out[probe::kTimes + n++] = v;
            }
            out[probe::kNumTimes] = n;
        }
    };

    // Thread state for the probes: no address step, Dst offset 0, window format 0,
    // predication on with every lane enabled.
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

    // Predication back to every lane enabled (the state other vector code assumes).
    inline void finish() {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        tensix_sync();
    }

    inline void write_lanes(uint32_t a, const volatile uint32_t* src) {
        volatile uint32_t* w = win();
        for (uint32_t l = 0; l < 32; ++l) {
            w[lane_word(a, l)] = src[l];
        }
        order(lane_word(a, 31));
    }

    // Caller has synchronised (tensix_sync) after the stores to address a.
    inline void read_lanes(uint32_t a, volatile uint32_t* dst) {
        volatile uint32_t* w = win();
        for (uint32_t l = 0; l < 32; ++l) {
            dst[l] = w[lane_word(a, l)];
        }
    }

    // ---- Probe 0: Dst window round trip, lane ids, clock check -------------------

    inline void smoke(const volatile uint32_t* in, volatile uint32_t* out, Times& t) {
        write_lanes(0, in);
        write_lanes(2, in + 32);
        TTI_SFPLOAD(L0, kInt32, kAm, 0);
        TTI_SFPLOAD(L1, kInt32, kAm, 2);
        TTI_SFPMOV(0, LANE, L2, 0);
        TTI_SFPNOP;
        TTI_SFPSTORE(L0, kInt32, kAm, 8);
        TTI_SFPSTORE(L1, kInt32, kAm, 10);
        TTI_SFPSTORE(L2, kInt32, kAm, 12);
        tensix_sync();
        read_lanes(8, out + probe::kData);
        read_lanes(10, out + probe::kData + 32);
        read_lanes(12, out + probe::kData + 64);

        // Wall clock against mcycle over a spin, then the cost of an idle tensix_sync.
        const uint32_t w0 = wall(), c0 = mcycle();
        for (uint32_t i = 0; i < 100000; ++i) {
            asm volatile("nop");
        }
        const uint32_t w1 = wall(), c1 = mcycle();
        t.add(w1 - w0);
        t.add(c1 - c0);
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t a = wall();
            tensix_sync();
            const uint32_t b = wall();
            t.add(b - a);
        }
        for (uint32_t k = 0; k < 8; ++k) {
            const uint32_t a = wall();
            const uint32_t b = wall();
            t.add(b - a);
        }
    }

    // ---- Probe 1: cost of predication --------------------------------------------

#define TP_MAD(r) TTI_SFPMAD(r, ONE, ONE, r, 0)
#define TP_MAD4 \
    TP_MAD(L0); \
    TP_MAD(L1); \
    TP_MAD(L2); \
    TP_MAD(L3)
#define TP_CHAIN4 \
    TP_MAD(L0);   \
    TP_MAD(L0);   \
    TP_MAD(L0);   \
    TP_MAD(L0)
#define TP_X8(m) \
    m;           \
    m;           \
    m;           \
    m;           \
    m;           \
    m;           \
    m;           \
    m
// Nesting level k: push the flags, then keep lanes with 2 lane - 2k >= 0.
#define TP_LEVEL(k)           \
    TTI_SFPPUSHC(0, 0, 0, 0); \
    TTI_SFPIADD((0x1000 - 2 * (k)) & 0xfff, LANE, L6, 1 | 8)
#define TP_NEST8 \
    TP_LEVEL(1); \
    TP_LEVEL(2); \
    TP_LEVEL(3); \
    TP_LEVEL(4); \
    TP_LEVEL(5); \
    TP_LEVEL(6); \
    TP_LEVEL(7); \
    TP_LEVEL(8)
#define TP_POP8 TP_X8(TTI_SFPPOPC(0, 0, 0, 0))

    enum Mask : uint32_t { kAll = 0, kOne = 1, kNone = 2, kNest8 = 3 };

    // L0-L3 = float(2 x lane) with every lane enabled.
    inline void pred_regs() {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPCAST(LANE, L4, 0);
        TTI_SFPNOP;
        TTI_SFPMOV(0, L4, L0, 0);
        TTI_SFPMOV(0, L4, L1, 0);
        TTI_SFPMOV(0, L4, L2, 0);
        TTI_SFPMOV(0, L4, L3, 0);
    }

    inline void pred_mask(Mask m) {
        if (m == kOne) {  // lane 13 only
            TTI_SFPIADD((0x1000 - 26) & 0xfff, LANE, L6, 1 | 4);
            TTI_SFPSETCC(0, L6, 0, 6);
        } else if (m == kNone) {
            TTI_SFPENCC(1, 0, 0, 10);
        } else if (m == kNest8) {
            TP_NEST8;
        }
    }

    inline void pred_unmask(Mask m) {
        if (m == kNest8) {
            TP_POP8;
        }
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
    }

    // K timed runs of `reps` replays of the first `len` buffer entries under mask m;
    // the registers of the last run go to Dst rows 64 slot .. 64 slot + 15.
    [[gnu::noinline]] void pred_run(
        Mask m,
        uint32_t reps,
        uint32_t k_runs,
        uint32_t slot,
        Times& t,
        uint32_t len = 32
    ) {
        for (uint32_t k = 0; k < k_runs; ++k) {
            pred_regs();
            pred_mask(m);
            tensix_sync();
            const uint32_t a = wall();
            const uint32_t replay = TT_OP_REPLAY(0, len, 0, 0);
            for (uint32_t r = 0; r < reps; ++r) {
                instrn_buffer[0] = replay;
            }
            tensix_sync();
            const uint32_t b = wall();
            t.add(b - a);
            pred_unmask(m);
        }
        const uint32_t base = 64 * slot;
        TT_SFPSTORE(L0, kInt32, kAm, base);
        TT_SFPSTORE(L1, kInt32, kAm, base + 4);
        TT_SFPSTORE(L2, kInt32, kAm, base + 8);
        TT_SFPSTORE(L3, kInt32, kAm, base + 12);
        tensix_sync();
    }

    // Timing: K runs per variant (empty, all, one, none, nest8, chain, nest-cost).
    // Data: per variant (all, one, none, nest8, chain) L0-L3 x 32 lanes.
    inline void pred(volatile uint32_t* out, uint32_t reps, uint32_t k_runs, Times& t) {
        TTI_REPLAY(0, 32, 0, 1);
        TP_X8(TP_MAD4);
        pred_run(kAll, 0, k_runs, 7, t);
        pred_run(kAll, reps, k_runs, 0, t);
        pred_run(kOne, reps, k_runs, 1, t);
        pred_run(kNone, reps, k_runs, 2, t);
        pred_run(kNest8, reps, k_runs, 3, t);
        TTI_REPLAY(0, 32, 0, 1);
        TP_X8(TP_CHAIN4);
        pred_run(kAll, reps, k_runs, 4, t);
        // 8 levels entered and left (24 instructions), flags restored each time.
        TTI_REPLAY(0, 24, 0, 1);
        TP_NEST8;
        TP_POP8;
        pred_run(kAll, reps, k_runs, 5, t, 24);
        for (uint32_t v = 0; v < 6; ++v) {
            for (uint32_t r = 0; r < 4; ++r) {
                read_lanes(64 * v + 4 * r, out + probe::kData + 128 * v + 32 * r);
            }
        }
    }

    // ---- Probe 2: the "any lane enabled" test ------------------------------------

    // L0 = 2 lane (counter), L1 = 0 (iterations seen), every lane enabled.
    inline void cnt_init() {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMOV(0, LANE, L0, 0);
        TTI_SFPLOADI(L1, 2, 0);
        TTI_SFPNOP;
    }

    // One body: L1 += 1; L0 -= 2, lane stays enabled while L0 >= 0. Broadcast i is
    // disabled by body i + 1.
    inline void cnt_body() {
        TTI_SFPIADD(1, L1, L1, 1 | 4);
        TTI_SFPIADD(0xffe, L0, L0, 1 | 8);
    }

    inline uint32_t any_bit(uint32_t q) {
        return (q >> 11) & 1;
    }

    // Polls this thread's SFPU busy bit until clear; returns the polls (0: was clear).
    inline uint32_t wait_sfpu_idle() {
        uint32_t n = 0;
        while (((bstatus_raw() >> 9) & 1) && n < probe::kSpinLimit) {
            ++n;
        }
        return n;
    }

    // Stores L1 (iterations per lane) to Dst at a with every lane enabled.
    inline void cnt_store(uint32_t a) {
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        TT_SFPSTORE(L1, kInt32, kAm, a);
        tensix_sync();
    }

    enum Check : uint32_t {
        kSync = 0,       // tensix_sync, then the CSR
        kBusy = 1,       // poll the SFPU busy bit, then the CSR
        kFence = 2,      // fence, then the CSR
        kOrWindow = 3,   // counters to Dst, the core reads 32 lanes and ORs
        kFenceBusy = 4,  // fence, poll the busy bit, then the CSR
    };

    constexpr uint32_t kChecks = 5;

    // "Any lane still enabled" after the body that was just pushed, by method c.
    inline uint32_t any_check(Check c, uint32_t& err) {
        if (c == kSync) {
            tensix_sync();
            return any_bit(qstatus_raw());
        }
        if (c == kFence || c == kFenceBusy) {
            asm volatile("fence" ::: "memory");
        }
        if (c == kBusy || c == kFenceBusy) {
            if (wait_sfpu_idle() >= probe::kSpinLimit) {
                err |= probe::kErrSpin;
            }
        }
        if (c != kOrWindow) {
            return any_bit(qstatus_raw());
        }
        // kOrWindow: L0 of every lane to Dst row 200, the core ORs "L0 >= 0".
        TTI_SFPPUSHC(0, 0, 0, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
        TTI_SFPSTORE(L0, kInt32, kAm, 200);
        TTI_SFPPOPC(0, 0, 0, 0);
        tensix_sync();
        volatile uint32_t* w = win();
        uint32_t any = 0;
        for (uint32_t l = 0; l < 32; ++l) {
            any |= ~w[lane_word(200, l)];
        }
        return any >> 31;
    }

    // Loop "body while any lane enabled" (bound 64): K runs, timing word and trip
    // count per run; L1 of the last run to Dst at a.
    [[gnu::noinline]] void any_loop(
        Check c,
        uint32_t k_runs,
        uint32_t a,
        volatile uint32_t* trips,
        Times& t,
        uint32_t& err
    ) {
        for (uint32_t k = 0; k < k_runs; ++k) {
            cnt_init();
            tensix_sync();
            const uint32_t t0 = wall();
            uint32_t n = 0, any = 1;
            while (any && n < 64) {
                cnt_body();
                ++n;
                any = any_check(c, err);
            }
            const uint32_t t1 = wall();
            t.add(t1 - t0);
            trips[k] = n;
        }
        cnt_store(a);
    }

    constexpr uint32_t kDelays[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24};
    constexpr uint32_t kNumDelays = sizeof(kDelays) / sizeof(kDelays[0]);

    // Data words: see "Output words" in README.md.
    inline void anycc(
        volatile uint32_t* out, uint32_t k_runs, Times& t, uint32_t& err
    ) {
        volatile uint32_t* d = out + probe::kData;
        // Recording: 40 bodies; after each, qstatus with no wait, after a fence, after
        // the busy bit clears, after tensix_sync. Word = 4 bits | bit 24 | polls << 8.
        for (uint32_t pass = 0; pass < 3; ++pass) {
            cnt_init();
            tensix_sync();
            for (uint32_t k = 0; k < 40; ++k) {
                cnt_body();
                const uint32_t q0 = qstatus_raw();
                asm volatile("fence" ::: "memory");
                const uint32_t q1 = qstatus_raw();
                const uint32_t polls = wait_sfpu_idle();
                const uint32_t q2 = qstatus_raw();
                tensix_sync();
                const uint32_t q3 = qstatus_raw();
                d[40 * pass + k] =
                    (any_bit(q0) | any_bit(q1) << 1 | any_bit(q2) << 2 |
                     any_bit(q3) << 3 | ((q3 >> 24) & 1) << 4 |
                     (polls > 0xffff ? 0xffff : polls) << 8);
            }
        }
        // Delay sweep: fence, wait kDelays[i] cycles (mcycle), read the bit.
        for (uint32_t i = 0; i < kNumDelays; ++i) {
            cnt_init();
            tensix_sync();
            for (uint32_t k = 0; k < 40; ++k) {
                cnt_body();
                asm volatile("fence" ::: "memory");
                const uint32_t c0 = mcycle();
                while (mcycle() - c0 < kDelays[i]) {
                    ;
                }
                d[512 + 40 * i + k] = any_bit(qstatus_raw());
                tensix_sync();
            }
        }
        // Loops ended by each check method; then the same 32 bodies with no check.
        for (uint32_t m = 0; m < kChecks; ++m) {
            any_loop(
                static_cast<Check>(m), k_runs, 256 + 4 * m, d + 128 + k_runs * m, t, err
            );
        }
        for (uint32_t k = 0; k < k_runs; ++k) {
            cnt_init();
            tensix_sync();
            const uint32_t t0 = wall();
            for (uint32_t n = 0; n < 32; ++n) {
                cnt_body();
            }
            tensix_sync();
            const uint32_t t1 = wall();
            t.add(t1 - t0);
        }
        cnt_store(256 + 4 * kChecks);
        // Cost of one check alone, per method, with every lane disabled.
        cnt_init();
        for (uint32_t n = 0; n < 32; ++n) {
            cnt_body();
        }
        for (uint32_t c = 0; c < kChecks; ++c) {
            for (uint32_t k = 0; k < k_runs; ++k) {
                tensix_sync();
                const uint32_t t0 = wall();
                const uint32_t any = any_check(static_cast<Check>(c), err);
                const uint32_t t1 = wall();
                t.add(t1 - t0);
                d[160 + c] = any;
            }
        }
        for (uint32_t v = 0; v <= kChecks; ++v) {
            read_lanes(256 + 4 * v, d + 256 + 32 * v);
        }
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPNOP;
    }

    // ---- Probe 3: gather by copy through the Dst window --------------------------

    // One record per lane: L_w = word w of record idx[lane], for w < W.
    template <uint32_t W>
    [[gnu::noinline]] void gather_step(
        const volatile uint32_t* table, uint32_t* idx, Times& t
    ) {
        volatile uint32_t* w = win();
        const uint32_t t0 = wall();
        TTI_SFPSTORE(L7, kInt32, kAm, 132);
        tensix_sync();
        for (uint32_t l = 0; l < 32; ++l) {
            idx[l] = w[lane_word(132, l)];
        }
        const uint32_t t1 = wall();
        for (uint32_t l = 0; l < 32; ++l) {
            const volatile uint32_t* rec = table + 4 * (idx[l] & 255);
#pragma GCC unroll 4
            for (uint32_t k = 0; k < W; ++k) {
                w[lane_word(4 * k, l)] = rec[k];
            }
        }
        order(lane_word(4 * (W - 1), 31));
        const uint32_t t2 = wall();
        TTI_SFPLOAD(L0, kInt32, kAm, 0);
        if constexpr (W > 1) {
            TTI_SFPLOAD(L1, kInt32, kAm, 4);
        }
        if constexpr (W > 2) {
            TTI_SFPLOAD(L2, kInt32, kAm, 8);
            TTI_SFPLOAD(L3, kInt32, kAm, 12);
        }
        tensix_sync();
        const uint32_t t3 = wall();
        t.add(t1 - t0);
        t.add(t2 - t1);
        t.add(t3 - t2);
    }

    // The same step with the loads grouped ahead of the stores that use them: 8
    // window loads per group for the indices, two records per group for the copy.
    template <uint32_t W>
    [[gnu::noinline]] void gather_step_fast(const uint32_t* table, Times& t) {
        volatile uint32_t* w = win();
        uint32_t idx[32];
        const uint32_t t0 = wall();
        TTI_SFPSTORE(L7, kInt32, kAm, 132);
        tensix_sync();
#pragma GCC unroll 4
        for (uint32_t g = 0; g < 32; g += 8) {
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                v[j] = w[lane_word(132, g + j)];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                idx[g + j] = v[j] & 255;
            }
        }
        const uint32_t t1 = wall();
#pragma GCC unroll 2
        for (uint32_t l = 0; l < 32; l += 2) {
            const uint32_t* ra = table + 4 * idx[l];
            const uint32_t* rb = table + 4 * idx[l + 1];
            uint32_t a[W], b[W];
#pragma GCC unroll 4
            for (uint32_t k = 0; k < W; ++k) {
                a[k] = ra[k];
                b[k] = rb[k];
            }
#pragma GCC unroll 4
            for (uint32_t k = 0; k < W; ++k) {
                w[lane_word(4 * k, l)] = a[k];
                w[lane_word(4 * k, l + 1)] = b[k];
            }
        }
        order(lane_word(4 * (W - 1), 31));
        const uint32_t t2 = wall();
        TTI_SFPLOAD(L0, kInt32, kAm, 0);
        if constexpr (W > 1) {
            TTI_SFPLOAD(L1, kInt32, kAm, 4);
        }
        if constexpr (W > 2) {
            TTI_SFPLOAD(L2, kInt32, kAm, 8);
            TTI_SFPLOAD(L3, kInt32, kAm, 12);
        }
        tensix_sync();
        const uint32_t t3 = wall();
        t.add(t1 - t0);
        t.add(t2 - t1);
        t.add(t3 - t2);
    }

#if defined(__riscv_vector)
    // The same step on T2's vector unit (lane l of Dst address a is window byte
    // 64 a + 8 l): indices by one stride-8 load, each record word by vluxei32 + store.
    template <uint32_t W>
    [[gnu::noinline]] void gather_step_rvv(const uint32_t* table, Times& t) {
        auto* w = reinterpret_cast<uint32_t*>(kDstWindow);
        const uint32_t t0 = wall();
        TTI_SFPSTORE(L7, kInt32, kAm, 132);
        tensix_sync();
        const size_t vl = __riscv_vsetvl_e32m8(32);
        vuint32m8_t idx = __riscv_vlse32_v_u32m8(w + 16 * 132, 8, vl);
        vuint32m8_t off = (
            __riscv_vsll_vx_u32m8(__riscv_vand_vx_u32m8(idx, 255, vl), 4, vl)
        );
        asm volatile("" ::: "memory");
        const uint32_t t1 = wall();
#pragma GCC unroll 4
        for (uint32_t k = 0; k < W; ++k) {
            vuint32m8_t v = __riscv_vluxei32_v_u32m8(table + k, off, vl);
            __riscv_vsse32_v_u32m8(w + 16 * 4 * k, 8, v, vl);
        }
        asm volatile("" ::: "memory");
        order(lane_word(4 * (W - 1), 31));
        const uint32_t t2 = wall();
        TTI_SFPLOAD(L0, kInt32, kAm, 0);
        if constexpr (W > 1) {
            TTI_SFPLOAD(L1, kInt32, kAm, 4);
        }
        if constexpr (W > 2) {
            TTI_SFPLOAD(L2, kInt32, kAm, 8);
            TTI_SFPLOAD(L3, kInt32, kAm, 12);
        }
        tensix_sync();
        const uint32_t t3 = wall();
        t.add(t1 - t0);
        t.add(t2 - t1);
        t.add(t3 - t2);
    }
#endif

    // Input: table [0, 1024) (256 records x 4 words), index sets [1024 + 32 s + l).
    // Data: for the last n_sets reps, set s: L_w lane l at 128 s + 32 w + l.
    template <uint32_t W>
    void gather(
        const volatile uint32_t* in,
        volatile uint32_t* out,
        uint32_t reps,
        uint32_t n_sets,
        uint32_t variant,
        Times& t
    ) {
        uint32_t idx[32];
        asm volatile("fence" ::: "memory");  // no stale L0 lines of the table
        for (uint32_t r = 0; r < reps; ++r) {
            const uint32_t s = r % n_sets;
            write_lanes(128, in + 1024 + 32 * s);
            TTI_SFPLOAD(L7, kInt32, kAm, 128);
            tensix_sync();
            if (variant == 1) {
                gather_step_fast<W>(const_cast<const uint32_t*>(in), t);
            } else if (variant == 2) {
#if defined(__riscv_vector)
                gather_step_rvv<W>(const_cast<const uint32_t*>(in), t);
#endif
            } else {
                gather_step<W>(in, idx, t);
            }
            if (r + n_sets >= reps) {
                TTI_SFPSTORE(L0, kInt32, kAm, 256);
                TTI_SFPSTORE(L1, kInt32, kAm, 260);
                TTI_SFPSTORE(L2, kInt32, kAm, 264);
                TTI_SFPSTORE(L3, kInt32, kAm, 268);
                tensix_sync();
                for (uint32_t k = 0; k < W; ++k) {
                    read_lanes(256 + 4 * k, out + probe::kData + 128 * s + 32 * k);
                }
            }
        }
    }

    // ---- Probe 4: one node against 32 rays (slab test) ---------------------------

    // Rays in Dst: inv x y z at 0, 4, 8; ob = -o inv at 12, 16, 20; tmax at 24.
    // L7 = -1.0; per node L4 = tnear, L5 = tfar; L6 = tfar - tnear (hit: sign clear).
    inline void slab_begin() {
        TTI_SFPLOADI(L4, 0, 0);
        TTI_SFPLOAD(L5, kInt32, kAm, 24);
    }

    // Axis a with slab bounds in L0 (min plane) and L1 (max plane).
#define TP_SLAB_AXIS(a)                         \
    TTI_SFPLOAD(L2, kInt32, kAm, 4 * (a));      \
    TTI_SFPLOAD(L3, kInt32, kAm, 12 + 4 * (a)); \
    TTI_SFPMAD(L0, L2, L3, L0, 0);              \
    TTI_SFPMAD(L1, L2, L3, L1, 0);              \
    TTI_SFPNOP;                                 \
    TTI_SFPSWAP(0, L1, L0, 1);                  \
    TTI_SFPSWAP(0, L4, L0, 1);                  \
    TTI_SFPSWAP(0, L1, L5, 1)

    inline void slab_end(uint32_t n) {
        TTI_SFPMAD(L4, L7, L5, L6, 0);
        TT_SFPSTORE(L6, kInt32, kAm, 64 + 4 * n);
    }

    inline void load_f32(uint32_t lreg, uint32_t bits) {
        TT_SFPLOADI(lreg, 0, bits >> 16);
        TT_SFPLOADI(lreg, 10, bits & 0xffff);
    }

    // Node fields from L1 (bmin x y z, bmax x y z as fp32 bits), pushed as immediates.
    inline void slab_node_f32(const volatile uint32_t* node, uint32_t n) {
        slab_begin();
        load_f32(L0, node[0]);
        load_f32(L1, node[3]);
        TP_SLAB_AXIS(0);
        load_f32(L0, node[1]);
        load_f32(L1, node[4]);
        TP_SLAB_AXIS(1);
        load_f32(L0, node[2]);
        load_f32(L1, node[5]);
        TP_SLAB_AXIS(2);
        slab_end(n);
    }

    // The same with bf16 node fields (one immediate each; low 16 bits of the words).
    inline void slab_node_bf16(const volatile uint32_t* node, uint32_t n) {
        slab_begin();
        TT_SFPLOADI(L0, 0, node[0] & 0xffff);
        TT_SFPLOADI(L1, 0, node[3] & 0xffff);
        TP_SLAB_AXIS(0);
        TT_SFPLOADI(L0, 0, node[1] & 0xffff);
        TT_SFPLOADI(L1, 0, node[4] & 0xffff);
        TP_SLAB_AXIS(1);
        TT_SFPLOADI(L0, 0, node[2] & 0xffff);
        TT_SFPLOADI(L1, 0, node[5] & 0xffff);
        TP_SLAB_AXIS(2);
        slab_end(n);
    }

#ifndef SLAB_BOX
#define SLAB_BOX 0, 0, 0, 0, 0, 0
#endif

    // Node fixed at compile time (SLAB_BOX: six bf16 bit patterns): no core work.
    template <
        uint32_t X0,
        uint32_t Y0,
        uint32_t Z0,
        uint32_t X1,
        uint32_t Y1,
        uint32_t Z1>
    inline void slab_node_const(uint32_t n) {
        slab_begin();
        TTI_SFPLOADI(L0, 0, X0);
        TTI_SFPLOADI(L1, 0, X1);
        TP_SLAB_AXIS(0);
        TTI_SFPLOADI(L0, 0, Y0);
        TTI_SFPLOADI(L1, 0, Y1);
        TP_SLAB_AXIS(1);
        TTI_SFPLOADI(L0, 0, Z0);
        TTI_SFPLOADI(L1, 0, Z1);
        TP_SLAB_AXIS(2);
        slab_end(n);
    }

    enum SlabMode : uint32_t { kF32 = 0, kBf16 = 1, kConst = 2 };

    // Hit mask (bit l: lane l hit) of the node at Dst address a, read by the core: a
    // plain loop (0), loads grouped by 8 (1) or T2's vector unit (2: stride-8 load).
    inline uint32_t hit_mask(uint32_t a, uint32_t variant) {
        volatile uint32_t* w = win();
        if (variant == 0) {
            uint32_t miss = 0;
            for (uint32_t l = 0; l < 32; ++l) {
                miss |= (w[lane_word(a, l)] >> 31) << l;
            }
            return ~miss;
        }
#if defined(__riscv_vector)
        if (variant == 2) {
            uint32_t mask = 0;
            const size_t vl = __riscv_vsetvl_e32m8(32);
            vint32m8_t v = (
                __riscv_vlse32_v_i32m8(
                    reinterpret_cast<const int32_t*>(kDstWindow) + 16 * a, 8, vl
                )
            );
            __riscv_vsm_v_b4(
                reinterpret_cast<uint8_t*>(&mask),
                __riscv_vmsge_vx_i32m8_b4(v, 0, vl),
                vl
            );
            asm volatile("" ::: "memory");
            return mask;
        }
#endif
        uint32_t miss = 0;
#pragma GCC unroll 4
        for (uint32_t g = 0; g < 32; g += 8) {
            uint32_t v[8];
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                v[j] = w[lane_word(a, g + j)];
            }
#pragma GCC unroll 8
            for (uint32_t j = 0; j < 8; ++j) {
                miss |= (v[j] >> 31) << (g + j);
            }
        }
        return ~miss;
    }

    // Words in and out: see "Output words" in README.md.
    [[gnu::noinline]] void slab(
        const volatile uint32_t* in,
        volatile uint32_t* out,
        uint32_t n_nodes,
        uint32_t reps,
        SlabMode mode,
        uint32_t readout,
        Times& t
    ) {
        for (uint32_t k = 0; k < 7; ++k) {
            write_lanes(4 * k, in + 32 * k);
        }
        TTI_SFPLOADI(L7, 0, 0xBF80);
        const volatile uint32_t* nodes = in + 512;
        volatile uint32_t* d = out + probe::kData;
        for (uint32_t r = 0; r < reps; ++r) {
            tensix_sync();
            const uint32_t t0 = wall();
            for (uint32_t n = 0; n < n_nodes; ++n) {
                if (mode == kF32) {
                    slab_node_f32(nodes + 6 * n, n);
                } else if (mode == kBf16) {
                    slab_node_bf16(nodes + 6 * n, n);
                } else {
                    slab_node_const<SLAB_BOX>(n);
                }
            }
            tensix_sync();
            const uint32_t t1 = wall();
            for (uint32_t n = 0; n < n_nodes; ++n) {
                d[n] = hit_mask(64 + 4 * n, readout);
            }
            const uint32_t t2 = wall();
            t.add(t1 - t0);
            t.add(t2 - t1);
        }
        for (uint32_t n = 0; n < n_nodes; ++n) {
            read_lanes(64 + 4 * n, d + 64 + 32 * n);
        }
    }

}  // namespace vec
