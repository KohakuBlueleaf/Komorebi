// Probes 11 (matrix-unit issue cost, K Dst blocks in rotation) and 14 (ties of MVMUL,
// packer); probe 11 keeps the order of the dense matrix-product kernel on every thread.
#pragma once

#include <stdint.h>

#include "probe.hpp"

namespace fpu {

    constexpr uint32_t kWallClock = 0xFFB121F0;  // RISCV_DEBUG_REG_WALL_CLOCK_L
    constexpr uint32_t kDstWindow = 0xFFBD8000;
    constexpr uint32_t kRows = 128;  // Dst rows (32-bit view) the ties probe reads
    constexpr uint32_t kTileWords = 1024;  // one packed tile: Dst rows 0..63
    constexpr uint32_t kAmInc = ADDR_MOD_7;  // Dst counter one block on (matmul init uses 0..5)
    constexpr uint32_t kAmClr = ADDR_MOD_6;  // Dst counter back to block 0
    constexpr uint32_t kAm = kAmClr;         // probe 14: no step

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

    // Probe 14 unpack thread: face 0 of tile 0 into SrcA and SrcB, then its done word.
    // One face: a face is a hand-over of a Src bank and a Src has two banks.
    inline void unpack(volatile uint32_t* out, uint32_t nonce) {
#ifdef TRISC_UNPACK
        const ckernel::TensorShape one_face{16, 16, 1, 1};
        _llk_unpack_AB_init_<BroadcastType::NONE>(one_face);
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

    // Address modes: kAmInc steps the Dst counter by ROWS (a block), kAmClr sets it
    // back to zero (as the matmul's reset mode does); no source counter moves.
    template <uint32_t ROWS>
    inline void math_modes() {
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = ROWS},
            .fidelity = {.incr = 0, .clr = 0},
        }
            .set(kAmInc);
        addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 0},
        }
            .set(kAmClr);
    }

    // Math thread counters zero and 8-row blocks; the Dst offset stays the one the
    // section protocol set.
    inline void math_counters() {
        math_modes<8>();
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, 15);
        tensix_sync();
    }

    // Probe 14: the same with the Dst offset zero.
    inline void math_state() {
        TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);
        math_counters();
    }

    // Math thread state with the window in FP32 (probe 14).
    inline void math_setup() {
        math_state();
        configure_dest_access<ThreadId::MathThreadId>(DataFormat::Float32, true);
        tensix_sync();
    }

    // Probe 11 (rows 3 and 4): a config is a body of kLen words, word i on Dst block
    // i mod K, played kReplays times.
    enum Op : uint32_t { kMvmul, kElwmul, kElwadd, kMova2d, kMovb2d };

    constexpr uint32_t kNumConfigs = 15;
    constexpr uint32_t kReps = 3, kReplays = 64;

    // The Dst block comes from the counter and the address mode, as in the LLK: the
    // row field of the word is zero.
    template <Op OP, uint32_t AM>
    inline void emit() {
        if constexpr (OP == kMvmul) {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, AM, 0);
        } else if constexpr (OP == kElwmul) {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, 0, AM, 0);
        } else if constexpr (OP == kElwadd) {
            TTI_ELWADD(p_setrwc::CLR_NONE, 1, 0, AM, 0);
        } else if constexpr (OP == kMova2d) {
            TTI_MOVA2D(0, 0, AM, p_mova2d::MOV_8_ROWS, 0);
        } else {
            TTI_MOVB2D(0, 0, AM, p_movb2d::MOV_4_ROWS, 0);
        }
    }

    template <Op OP, uint32_t K, uint32_t I = 0, uint32_t N = (K == 5 ? 20 : 24)>
    inline void body() {
        if constexpr (I < N) {
            emit<OP, (I % K == K - 1) ? kAmClr : kAmInc>();
            body<OP, K, I + 1, N>();
        }
    }

    // d: 0..2 spans of kReplays replays, 3 body length.
    template <Op OP, uint32_t K>
    inline void measure(volatile uint32_t* d) {
        constexpr uint32_t len = K == 5 ? 20 : 24;
        math_modes<(OP == kMovb2d ? 4 : 8)>();
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
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, kAmClr, 0);
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

    // Math thread: the config, then the release of SrcA and SrcB the unpack thread
    // handed over (CLR_AB, as the eltwise kernels release).
    inline void issue(uint32_t c, volatile uint32_t* out) {
#ifdef TRISC_MATH
        run_config(c, out + probe::kData + 32 * c);
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, 15);
        tensix_sync();
#endif
    }

    // Math thread, before the first section: counters, and the clock's own cost (two
    // syncs around nothing) in the slot after the last config.
    inline void math_prepare(volatile uint32_t* out) {
#ifdef TRISC_MATH
        math_counters();
        volatile uint32_t* e = out + probe::kData + 32 * kNumConfigs;
        for (uint32_t r = 0; r < 3; ++r) {
            const uint32_t a = wall();
            tensix_sync();
            e[r] = wall() - a;
        }
#endif
    }

    // Pack thread: the section packed to the scratch buffer, its words counted into
    // config c's words 4 (nonzero) and 5..12 (four value, count pairs).
    inline void collect(
        volatile uint32_t* out, uint32_t c, const volatile uint32_t* tile
    ) {
#ifdef TRISC_PACK
        cb_reserve_back(probe::kCbScratch, 1);
        pack_tile(0, probe::kCbScratch);
        tensix_sync();
        volatile uint32_t* d = out + probe::kData + 32 * c;
        uint32_t nonzero = 0, nv = 0;
        uint32_t val[4] = {0, 0, 0, 0}, cnt[4] = {0, 0, 0, 0};
        for (uint32_t j = 0; j < kTileWords; ++j) {
            const uint32_t v = tile[j];
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
#endif
    }

    // All three threads, one Dst section per config, in the order of the dense kernel:
    // acquire, unpack hand-over, math and release, commit, wait, pack, release.
    inline void skeleton(
        volatile uint32_t* out,
        const volatile uint32_t* tile,
        uint32_t first,
        uint32_t count
    ) {
        matmul_block_init(probe::kCbIn, probe::kCbIn, 0, 1, 1, 1);
        math_prepare(out);
        for (uint32_t i = 0; i < count; ++i) {
            tile_regs_acquire();
            UNPACK((llk_unpack_AB_matmul(probe::kCbIn, probe::kCbIn, 0, 0, 1, 1, 1)));
            issue(first + i, out);
            tile_regs_commit();
            tile_regs_wait();
            collect(out, first + i, tile);
            tile_regs_release();
        }
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
