// Probes 11 (matrix-unit issue cost, K Dst blocks in rotation) and 14 (ties of MVMUL,
// packer): unpack fills SrcA and SrcB, the math thread runs, the pack thread waits.
#pragma once

#include <stdint.h>

#include "probe.hpp"

namespace fpu {

    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L
    constexpr uint32_t kDstWindow = 0xFFBD8000;
    constexpr uint32_t kRows = 128;       // Dst rows (32-bit view) the probes use
    constexpr uint32_t kAm = ADDR_MOD_1;  // no step on any counter

    inline uint32_t wall() {
        return *reinterpret_cast<volatile uint32_t*>(kWallClock);
    }

    inline volatile uint32_t* win() {
        return reinterpret_cast<volatile uint32_t*>(kDstWindow);
    }

    inline void fence() {
        asm volatile("fence" ::: "memory");
    }

    // Waits (bounded) until out[w] holds the nonce; false on a timeout.
    inline bool wait_word(volatile uint32_t* out, uint32_t w, uint32_t nonce) {
        for (uint32_t n = 0; n < probe::kSpinLimit; ++n) {
            fence();
            if (out[w] == nonce) {
                return true;
            }
        }
        return false;
    }

    inline void post_word(volatile uint32_t* out, uint32_t w, uint32_t nonce) {
        fence();
        out[w] = nonce;
        fence();
    }

    // Unpack thread: tile 0 of the input CB into SrcA and SrcB, then its done word.
    inline void unpack(volatile uint32_t* out, uint32_t nonce) {
#ifdef TRISC_UNPACK
        llk_unpack_AB_init<BroadcastType::NONE>(probe::kCbIn, probe::kCbIn);
        llk_unpack_AB(probe::kCbIn, probe::kCbIn, 0, 0);
        tensix_sync();
#endif
        post_word(out, probe::kDone0, nonce);
    }

    // Dst rows 0..kRows-1 of the window set to zero (math thread, between syncs).
    inline void zero_rows() {
        tensix_sync();
        volatile uint32_t* w = win();
        for (uint32_t i = 0; i < 16 * kRows; ++i) {
            w[i] = 0;
        }
        (void)w[16 * kRows - 1];
        fence();
        tensix_sync();
    }

    // Math thread state: no counter step in kAm, Dst offset 0, counters (and the
    // fidelity counter) zero, window in FP32.
    inline void math_setup() {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
            .fidelity = {.incr = 0, .clr = 0},
        }
            .set(kAm);
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
        configure_dest_access<ThreadId::MathThreadId>(DataFormat::Float32, true);
        tensix_sync();
    }

    inline void set_fidelity_base(uint32_t phase) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        cfg_reg_rmw_tensix<FIDELITY_BASE_Phase_RMW>(phase & 3);
        tensix_sync();
    }

    // ---- Probe 11: issue cost (rows 3 and 4) ----------------------------------------

    enum Op : uint32_t {
        kNone,
        kMvmul,
        kElwmul,
        kElwadd,
        kElwsub,
        kGapool,
        kGmpool,
        kMova2d,
        kMovb2d,
        kMovd2a,
        kMovd2b
    };

    struct Config {
        Op op;
        uint32_t k;      // blocks in rotation, block j at Dst row 8 j
        uint32_t phase;  // fidelity base
    };

    constexpr Config kConfigs[] = {
        {kNone, 1, 0},   {kMvmul, 1, 0},  {kMvmul, 2, 0},  {kMvmul, 3, 0},
        {kMvmul, 4, 0},  {kMvmul, 5, 0},  {kMvmul, 6, 0},  {kMvmul, 8, 0},
        {kMvmul, 8, 1},  {kMvmul, 8, 2},  {kMvmul, 8, 3},  {kElwmul, 1, 0},
        {kElwmul, 8, 0}, {kElwadd, 1, 0}, {kElwadd, 8, 0}, {kElwsub, 8, 0},
        {kGapool, 1, 0}, {kGapool, 8, 0}, {kGmpool, 1, 0}, {kGmpool, 8, 0},
        {kMova2d, 1, 0}, {kMova2d, 8, 0}, {kMovb2d, 8, 0}, {kMovd2a, 8, 0},
        {kMovd2b, 8, 0},
    };
    constexpr uint32_t kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);
    constexpr uint32_t kReps = 3, kReplays = 64;

    // The instruction word of op writing block j (Dst row 8 j; source rows 32 + 4 j).
    inline uint32_t word(Op op, uint32_t j) {
        const uint32_t d = (8 * j) & 0x3FF, s = (32 + 4 * j) & 0x3F;
        switch (op) {
            case kMvmul:
                return TT_OP_MVMUL(0, 0, kAm, d);
            case kElwmul:
                return TT_OP_ELWMUL(0, 0, 0, kAm, d);
            case kElwadd:
                return TT_OP_ELWADD(0, 1, 0, kAm, d);
            case kElwsub:
                return TT_OP_ELWSUB(0, 1, 0, kAm, d);
            case kGapool:
                return TT_OP_GAPOOL(0, p_gpool::DIM_16X16, kAm, p_gpool::INDEX_DIS, d);
            case kGmpool:
                return TT_OP_GMPOOL(0, p_gpool::DIM_16X16, kAm, p_gpool::INDEX_DIS, d);
            case kMova2d:
                return TT_OP_MOVA2D(0, 0, kAm, p_mova2d::MOV_8_ROWS, d);
            case kMovb2d:
                return TT_OP_MOVB2D(0, 0, kAm, p_movb2d::MOV_4_ROWS, d);
            case kMovd2a:
                return TT_OP_MOVD2A(0, s, kAm, p_movd2a::MOV_4_ROWS, d);
            case kMovd2b:
                return TT_OP_MOVD2B(0, s, kAm, p_movd2b::MOV_4_ROWS, d);
            default:
                return TT_OP_NOP;
        }
    }

    // Config c at out[kData + 32 c]: 0..2 spans of kReplays replays of the body, 3 its
    // length, 4 nonzero Dst words (rows 0..127), 5..12 four (value, count) pairs.
    inline void issue(volatile uint32_t* out) {
        math_setup();
        for (uint32_t c = 0; c < kNumConfigs; ++c) {
            const Config cf = kConfigs[c];
            volatile uint32_t* d = out + probe::kData + 32 * c;
            const uint32_t len = cf.k == 5 ? 20 : 24;
            zero_rows();
            set_fidelity_base(cf.phase);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
            instrn_buffer[0] = TT_OP_REPLAY(0, len, 0, 1);  // record the next len
            for (uint32_t i = 0; i < len; ++i) {
                instrn_buffer[0] = word(cf.op, i % cf.k);
            }
            const uint32_t rep = TT_OP_REPLAY(0, len, 0, 0);
            tensix_sync();
            for (uint32_t r = 0; r < kReps; ++r) {
                const uint32_t a = wall();
                if (cf.op != kNone) {
                    for (uint32_t i = 0; i < kReplays; ++i) {
                        instrn_buffer[0] = rep;
                    }
                }
                tensix_sync();
                d[r] = wall() - a;
            }
            d[3] = len;
            // Histogram of the nonzero words.
            volatile uint32_t* w = win();
            uint32_t nonzero = 0, nv = 0;
            uint32_t val[4] = {0, 0, 0, 0}, cnt[4] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < 16 * kRows; ++i) {
                const uint32_t v = w[i];
                if (v == 0) {
                    continue;
                }
                ++nonzero;
                uint32_t q = 0;
                while (q < nv && val[q] != v) {
                    ++q;
                }
                if (q == nv && nv < 4) {
                    val[nv++] = v;
                }
                if (q < nv) {
                    ++cnt[q];
                }
            }
            d[4] = nonzero;
            for (uint32_t q = 0; q < 4; ++q) {
                d[5 + 2 * q] = val[q];
                d[6 + 2 * q] = cnt[q];
            }
        }
        set_fidelity_base(0);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 15);
        tensix_sync();
    }

    // ---- Probe 14: ties (row 7) -----------------------------------------------------

    constexpr uint32_t kSrBlocks = 7;  // stochastic-rounding products at rows 8 .. 63

    inline void set_fpu_srnd(uint32_t on) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        cfg_reg_rmw_tensix<ALU_ROUNDING_MODE_Fpu_srnd_en_RMW>(on & 1);
        tensix_sync();
    }

    // Math thread: Dst rows 0..7 = SrcB @ SrcA, rounded to nearest; rows 8 j (j = 1..7)
    // the same with stochastic rounding; then commit the section.
    inline void ties_math() {
        tile_regs_acquire();
        math_setup();
        zero_rows();
        TTI_MVMUL(0, 0, kAm, 0);
        tensix_sync();
        set_fpu_srnd(1);
        for (uint32_t j = 1; j <= kSrBlocks; ++j) {
            instrn_buffer[0] = TT_OP_MVMUL(0, 0, kAm, (8 * j) & 0x3FF);
        }
        tensix_sync();
        set_fpu_srnd(0);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 15);
        tensix_sync();
        tile_regs_commit();
    }

    inline void set_pack_srnd(uint32_t on) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<ALU_ROUNDING_MODE_Gasket_srnd_en_RMW>(on & 1);
        cfg_reg_rmw_tensix<ALU_ROUNDING_MODE_Packer_srnd_en_RMW>(on & 1);
        tensix_sync();
    }

    // Data 0..1023 = Dst rows 0..63 (products); tile 1 = in[1024 ..], packed to BF16
    // into cb_rne (data 1024..1535), then stochastically rounded into cb_sr (1536..).
    inline void ties_pack(
        const volatile uint32_t* in,
        volatile uint32_t* out,
        uint32_t cb_rne,
        uint32_t cb_sr,
        const volatile uint32_t* rne,
        const volatile uint32_t* sr
    ) {
        volatile uint32_t* d = out + probe::kData;
        tile_regs_wait();
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
        tensix_sync();
        volatile uint32_t* w = win();
        for (uint32_t i = 0; i < 1024; ++i) {
            d[i] = w[i];
        }
        for (uint32_t i = 0; i < 1024; ++i) {
            w[1024 + i] = in[1024 + i];
        }
        (void)w[2047];
        fence();
        tensix_sync();
        cb_reserve_back(cb_rne, 1);
        cb_reserve_back(cb_sr, 1);
        pack_tile(1, cb_rne);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // the packer is done
        tensix_sync();
        set_pack_srnd(1);
        pack_tile(1, cb_sr);
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
        tensix_sync();
        set_pack_srnd(0);
        tile_regs_release();
        fence();
        for (uint32_t i = 0; i < 512; ++i) {
            d[1024 + i] = rne[i];
            d[1536 + i] = sr[i];
        }
    }

}  // namespace fpu
