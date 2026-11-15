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

    // Dst rows 0..kRows-1 of the window set to zero (pack thread, between syncs).
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
    // fidelity counter) zero. No window access: Dst is read and written by the pack
    // thread.
    inline void math_state() {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0},
            .fidelity = {.incr = 0, .clr = 0},
        }
            .set(kAm);
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
        tensix_sync();
    }

    // Math thread state with the window in FP32 (probe 14).
    inline void math_setup() {
        math_state();
        configure_dest_access<ThreadId::MathThreadId>(DataFormat::Float32, true);
        tensix_sync();
    }

    // ---- Probe 11: issue cost (rows 3 and 4) ----------------------------------------
    //
    // Each config is a body of kLen instruction words (kLen - 4 when K = 5), word i
    // writing Dst block i mod K (rows 8 (i mod K)), recorded once into the replay
    // buffer with TTI_ words (the form every replay in the LLK uses) and played
    // kReplays times a span. The math thread only issues; the pack thread zeroes Dst
    // before each config and reads it after it, so no window access meets a running
    // matrix-unit instruction. Config 0 is one MVMUL, pushed once.

    enum Op : uint32_t { kMvmul, kElwmul, kElwadd, kMova2d, kMovb2d };

    constexpr uint32_t kNumConfigs = 15;
    constexpr uint32_t kReps = 3, kReplays = 64;
    constexpr uint32_t kSeqMath = 12;  // math thread: config i done (nonce + 1 + i)
    constexpr uint32_t kSeqPack = 13;  // pack thread: Dst zero again (nonce + i)

    template <Op OP, uint32_t D>
    inline void emit() {
        if constexpr (OP == kMvmul) {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, kAm, D);
        } else if constexpr (OP == kElwmul) {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, 0, kAm, D);
        } else if constexpr (OP == kElwadd) {
            TTI_ELWADD(p_setrwc::CLR_NONE, 1, 0, kAm, D);
        } else if constexpr (OP == kMova2d) {
            TTI_MOVA2D(0, 0, kAm, p_mova2d::MOV_8_ROWS, D);
        } else {
            TTI_MOVB2D(0, 0, kAm, p_movb2d::MOV_4_ROWS, D);
        }
    }

    template <Op OP, uint32_t K, uint32_t I = 0, uint32_t N = (K == 5 ? 20 : 24)>
    inline void body() {
        if constexpr (I < N) {
            emit<OP, 8 * (I % K)>();
            body<OP, K, I + 1, N>();
        }
    }

    // d: 0..2 spans of kReplays replays, 3 body length.
    template <Op OP, uint32_t K>
    inline void measure(volatile uint32_t* d) {
        constexpr uint32_t len = K == 5 ? 20 : 24;
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
        TTI_REPLAY(0, len, 0, 1);
        body<OP, K>();
        const uint32_t rep = TT_OP_REPLAY(0, len, 0, 0);
        tensix_sync();
        for (uint32_t r = 0; r < kReps; ++r) {
            const uint32_t a = wall();
            for (uint32_t i = 0; i < kReplays; ++i) {
                instrn_buffer[0] = rep;
            }
            tensix_sync();
            d[r] = wall() - a;
        }
        d[3] = len;
    }

    // Config 0: one MVMUL, once.
    inline void single(volatile uint32_t* d) {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
        const uint32_t a = wall();
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, kAm, 0);
        tensix_sync();
        d[0] = wall() - a;
        d[3] = 1;
    }

    inline void run_config(uint32_t c, volatile uint32_t* d) {
        switch (c) {
            case 0:
                single(d);
                break;
            case 1:
                measure<kMvmul, 1>(d);
                break;
            case 2:
                measure<kMvmul, 2>(d);
                break;
            case 3:
                measure<kMvmul, 3>(d);
                break;
            case 4:
                measure<kMvmul, 4>(d);
                break;
            case 5:
                measure<kMvmul, 5>(d);
                break;
            case 6:
                measure<kMvmul, 6>(d);
                break;
            case 7:
                measure<kMvmul, 8>(d);
                break;
            case 8:
                measure<kElwmul, 1>(d);
                break;
            case 9:
                measure<kElwmul, 8>(d);
                break;
            case 10:
                measure<kElwadd, 1>(d);
                break;
            case 11:
                measure<kElwadd, 8>(d);
                break;
            case 12:
                measure<kMova2d, 1>(d);
                break;
            case 13:
                measure<kMova2d, 8>(d);
                break;
            default:
                measure<kMovb2d, 8>(d);
                break;
        }
    }

    // Math thread: configs first .. first + count - 1, each after the pack thread has
    // zeroed Dst; the empty span (two syncs around nothing) goes to the slot after the
    // last config. Every wait is bounded.
    inline void math(volatile uint32_t* out, uint32_t nonce, uint32_t first, uint32_t count) {
#ifdef TRISC_MATH
        if (!wait_word(out, probe::kDone0, nonce) || !wait_word(out, kSeqPack, nonce)) {
            out[probe::kAuxError] = probe::kErrorSpin;
            return;
        }
        math_state();
        volatile uint32_t* e = out + probe::kData + 32 * kNumConfigs;
        for (uint32_t r = 0; r < 3; ++r) {
            const uint32_t a = wall();
            tensix_sync();
            e[r] = wall() - a;
        }
        for (uint32_t i = 0; i < count; ++i) {
            run_config(first + i, out + probe::kData + 32 * (first + i));
            post_word(out, kSeqMath, nonce + 1 + i);
            if (!wait_word(out, kSeqPack, nonce + 1 + i)) {
                out[probe::kAuxError] = probe::kErrorSpin;
                break;
            }
        }
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 15);
        tensix_sync();
#endif
    }

    // Pack thread: zero Dst, then after each config of the math thread read rows
    // 0..kRows-1 into config c's words (4 nonzero words, 5..12 four (value, count)
    // pairs), zero them again and say so. Returns error bits.
    inline uint32_t watch(volatile uint32_t* out, uint32_t nonce, uint32_t first, uint32_t count) {
#ifdef TRISC_PACK
        configure_dest_access<ThreadId::PackThreadId>(DataFormat::Float32, true);
        tensix_sync();
        zero_rows();
        post_word(out, kSeqPack, nonce);
        for (uint32_t i = 0; i < count; ++i) {
            volatile uint32_t* d = out + probe::kData + 32 * (first + i);
            if (!wait_word(out, kSeqMath, nonce + 1 + i)) {
                return probe::kErrorSpin;
            }
            tensix_sync();
            volatile uint32_t* w = win();
            uint32_t nonzero = 0, nv = 0;
            uint32_t val[4] = {0, 0, 0, 0}, cnt[4] = {0, 0, 0, 0};
            for (uint32_t j = 0; j < 16 * kRows; ++j) {
                const uint32_t v = w[j];
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
            zero_rows();
            post_word(out, kSeqPack, nonce + 1 + i);
        }
#endif
        return 0;
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
