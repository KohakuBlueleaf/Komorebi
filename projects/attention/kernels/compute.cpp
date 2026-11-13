// Attention compute, the body of Kohaku-TT's ktt_flash compute (Apache-2.0): per job,
// up to R query tile rows against the job's key blocks taken from the lane buffer.
#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "attention.h"
#include "komorebi/device/km_lane.hpp"
#include "komorebi/device/km_worker.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu.hpp"
#endif

namespace {

    constexpr uint32_t DT = get_compile_time_arg_val(0);
    constexpr uint32_t DV = get_compile_time_arg_val(1);
    constexpr uint32_t R = get_compile_time_arg_val(2);
    constexpr uint32_t KB = get_compile_time_arg_val(3);
    constexpr uint32_t A_BITS = get_compile_time_arg_val(4);
    constexpr uint32_t THR_BITS = get_compile_time_arg_val(5);
    // Accumulator columns per row: O, then the row sum l (column DV).
    constexpr uint32_t W = DV + 1;
    constexpr uint32_t SC = kSubChunk;
    constexpr uint32_t ANY_WORD = 257;  // fp32 state tile: row 0, column 17 (the flag)
    constexpr uint32_t P = DT * KB + KB * DV;  // pages of a K / V block
    constexpr uint32_t V0 = DT * KB;           // its first V page

    constexpr uint32_t cb_q = kCbQ, cb_kv = kCbKv, cb_e = kCbOnes, cb_p = kCbP;
    constexpr uint32_t cb_st = kCbSt, cb_o = kCbO, cb_stok = kCbStok, cb_otok = kCbOtok;
    constexpr uint32_t cb_m = kCbCausal, cb_out = kCbOut;

    static_assert(KB % SC == 0, "key block must be whole sub-chunks");

    struct Step {
        uint32_t r;   // query tile row
        uint32_t c;   // chunk index along the keys (0 = first)
        uint32_t k0;  // first key tile of the chunk within the block
        uint32_t nt;  // key tiles in the chunk
        bool last;    // the row's last chunk
        bool diag;    // the chunk's last tile is the row's diagonal tile (causal mask)
    };

    uint32_t slot(uint32_t r, uint32_t c) {
        return 2 * r + (c & 1);
    }

    // Math thread: lane-wise row maxima of the section.
    void state_vector(uint32_t nt) {
#ifdef TRISC_MATH
        if (nt == 3) {
            flash::state_max<3>();
        } else if (nt == 2) {
            flash::state_max<2>();
        } else {
            flash::state_max<1>();
        }
#endif
    }

    // Pack thread: row state update and P of the section.
    void exp_vector(bool first, uint32_t nt) {
#ifdef TRISC_PACK
        flash::exp_section_begin();
        if (first) {
            flash::state_update<true>(THR_BITS);
        } else {
            flash::state_update<false>(THR_BITS);
        }
        if (nt == 3) {
            flash::exp_step<3>();
        } else if (nt == 2) {
            flash::exp_step<2>();
        } else {
            flash::exp_step<1>();
        }
        flash::exp_section_end();
#endif
    }

    // Math thread: vector work after the section's matmuls, holding the SFPU mutex.
    void math_vector_begin() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH)));
        MATH((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
        MATH((t6_mutex_acquire(mutex::SFPU)));
    }

    void math_vector_end() {
        MATH((TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::WAIT_SFPU)));
        MATH((t6_mutex_release(mutex::SFPU)));
    }

    // Packer output format (fp32: cb_o, cb_st; bf16: cb_p, cb_out) and L1
    // accumulation, changed only on transitions.
    bool pack_fp32 = false;
    bool pack_acc = false;

    void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(fp32 ? cb_p : cb_o, fp32 ? cb_o : cb_p);
            pack_fp32 = fp32;
        }
    }

    void pack_accumulate(bool acc) {
        if (acc != pack_acc) {
            pack_reconfig_l1_acc(acc ? 1 : 0);
            pack_acc = acc;
        }
    }

    // Pack thread: vector work waits for the math section and holds the SFPU mutex;
    // packs wait for the vector work.
    void vector_begin() {
        PACK((TTI_SEMWAIT(
            p_stall::STALL_SFPU | p_stall::STALL_SYNC,
            semaphore::t6_sem(semaphore::MATH_PACK),
            p_stall::STALL_ON_ZERO
        )));
        PACK((math::set_dst_write_addr<
              DstTileShape::Tile32x32,
              UnpackDestination::SrcRegs>(0)));
        PACK((t6_mutex_acquire(mutex::SFPU)));
    }

    void vector_end() {
        PACK((
            TTI_STALLWAIT(p_stall::STALL_SYNC | p_stall::STALL_PACK, p_stall::WAIT_SFPU)
        ));
        PACK((t6_mutex_release(mutex::SFPU)));
    }

    // S = Q_r K^T (raw scores) and the row state into Dst; state update and P on the
    // pack thread.
    void score(const Step& s) {
        const bool first = s.c == 0;
        tile_regs_acquire_math_clear();
        // First chunk: any slot (defines every state row).
        reconfig_data_format_srca(cb_kv, cb_st);
        copy_init(cb_st);
        copy_tile(cb_st, first ? slot(s.r, 1) : slot(s.r, s.c - 1), 3);
        reconfig_data_format_srca(cb_st, cb_kv);
        // The causal mask is copied into the diagonal score tile first; the score
        // matmul accumulates onto it.
#ifdef FLASH_CAUSAL
        if (s.diag) {
            copy_init(cb_m);
            copy_tile(cb_m, 0, s.nt - 1);
        }
#endif
        mm_no_mop_init_short(cb_q, cb_kv, true, s.nt, 1, DT);
        for (uint32_t d = 0; d < DT; ++d) {
            matmul_block_no_mop(
                cb_q, cb_kv, s.r * DT + d, d * KB + s.k0, 0, true, s.nt, 1, DT
            );
        }
        math_vector_begin();
        state_vector(s.nt);
        math_vector_end();
        tile_regs_commit();

        cb_reserve_back(cb_p, SC);
        cb_reserve_back(cb_stok, 1);
        tile_regs_wait();
        vector_begin();
        exp_vector(first, s.nt);
        vector_end();
        pack_format(false);
        pack_accumulate(false);
        for (uint32_t t = 0; t < s.nt; ++t) {
            pack_tile(t, cb_p);
        }
        pack_format(true);
        pack_tile<true>(3, cb_st, slot(s.r, s.c));
        tile_regs_release_math_clear();
        cb_push_back(cb_p, SC);
        cb_push_back(cb_stok, 1);
    }

    // Dst 0..: P V columns [j0, j0 + wv) of the block, then the row-sum column if
    // `sum`.
    void pv_columns(const Step& s, uint32_t j0, uint32_t wv, bool sum) {
        if (wv > 0) {
            mm_no_mop_init_short(cb_p, cb_kv, false, wv, 1, s.nt);
            for (uint32_t k = 0; k < s.nt; ++k) {
                matmul_block_no_mop(
                    cb_p, cb_kv, k, V0 + (s.k0 + k) * DV + j0, 0, false, wv, 1, s.nt
                );
            }
        }
        if (sum) {
            mm_no_mop_init_short(cb_p, cb_e, false, 1, 1, s.nt);
            for (uint32_t k = 0; k < s.nt; ++k) {
                matmul_block_no_mop(cb_p, cb_e, k, 0, wv, false, 1, 1, s.nt);
            }
        }
    }

    // O_r (and l_r) += P V (and P 1); scaled by alpha first when the tile's reference
    // moved.
    void output(const Step& s) {
        const bool first = s.c == 0;
        cb_wait_front(cb_stok, 1);
        cb_wait_front(cb_p, SC);
        const uint32_t moved = (
            first ? 0 : read_tile_value(cb_st, slot(s.r, s.c), ANY_WORD)
        );
        if (moved == 0) {
            for (uint32_t j0 = 0; j0 < W; j0 += 4) {
                const uint32_t w = W - j0 < 4 ? W - j0 : 4;
                const uint32_t wv = j0 < DV ? (DV - j0 < w ? DV - j0 : w) : 0;
                tile_regs_acquire_math_clear();
                pv_columns(s, j0, wv, j0 <= DV && DV < j0 + w);
                tile_regs_commit();
                tile_regs_wait();
                pack_format(true);
                pack_accumulate(!first);
                for (uint32_t t = 0; t < w; ++t) {
                    pack_tile<true>(t, cb_o, s.r * W + j0 + t);
                }
                tile_regs_release_math_clear();
            }
        } else {
            for (uint32_t j = 0; j < W; ++j) {
                tile_regs_acquire_math_clear();
                pv_columns(s, j, j < DV ? 1 : 0, j == DV);
                reconfig_data_format_srca(cb_kv, cb_st);
                copy_init(cb_st);
                copy_tile(cb_st, slot(s.r, s.c), 2);
                copy_tile(cb_st, slot(s.r, s.c - 1), 3);
                copy_init(cb_o);
                copy_tile(cb_o, s.r * W + j, 1);
                reconfig_data_format_srca(cb_o, cb_kv);
                tile_regs_commit();
                tile_regs_wait();
                vector_begin();
                PACK((flash::rescale_step()));
                vector_end();
                pack_format(true);
                pack_accumulate(false);
                pack_tile<true>(0, cb_o, s.r * W + j);
                tile_regs_release_math_clear();
            }
        }
        cb_pop_front(cb_p, SC);
        cb_pop_front(cb_stok, 1);
        if (s.last) {
            cb_reserve_back(cb_otok, 1);
            cb_push_back(cb_otok, 1);
        }
    }

    // Output row r = O_r / l_r (bf16), to the writer. Pack waits (bounded) for room;
    // without it the row is not packed and the record ends in a give-up.
    void finalize(const km::Compute& cp, uint32_t r) {
        cb_wait_front(cb_otok, 1);
        cb_pop_front(cb_otok, 1);
        const bool room = cp.reserve_back(cb_out, DV);
        for (uint32_t j0 = 0; j0 < DV; j0 += 3) {
            const uint32_t w = DV - j0 < 3 ? DV - j0 : 3;
            tile_regs_acquire_math_clear();
            reconfig_data_format_srca(cb_kv, cb_o);
            copy_init(cb_o);
            for (uint32_t t = 0; t < w; ++t) {
                copy_tile(cb_o, r * W + j0 + t, t);
            }
            copy_tile(cb_o, r * W + DV, 3);
            reconfig_data_format_srca(cb_o, cb_kv);
            tile_regs_commit();
            tile_regs_wait();
            vector_begin();
            PACK((flash::normalize_step(w)));
            vector_end();
            pack_format(false);
            pack_accumulate(false);
            for (uint32_t t = 0; room && t < w; ++t) {
                pack_tile(t, cb_out);
            }
            tile_regs_release_math_clear();
        }
        if (room) {
            cb_push_back(cb_out, DV);
        }
    }

    // The constant tiles, built by the vector unit and packed once: ones (cb_ones);
    // with FLASH_CAUSAL the diagonal mask (cb_causal).
    void constant_tiles() {
        tile_regs_acquire_math_clear();
        math_vector_begin();
        MATH((math::clear_dst_reg_addr()));
        MATH((flash::const_lanes()));
        MATH((flash::const_tile(flash::Const::Ones, 0, 0)));
#ifdef FLASH_CAUSAL
        MATH((flash::const_tile(flash::Const::Causal, 1, 0)));
#endif
        MATH((math::clear_dst_reg_addr()));
        math_vector_end();
        tile_regs_commit();
        tile_regs_wait();
        pack_format(false);
        pack_accumulate(false);
        cb_reserve_back(cb_e, 1);
        pack_tile(0, cb_e);
        cb_push_back(cb_e, 1);
#ifdef FLASH_CAUSAL
        cb_reserve_back(cb_m, 1);
        pack_tile(1, cb_m);
        cb_push_back(cb_m, 1);
#endif
        tile_regs_release_math_clear();
        cb_wait_front(cb_e, 1);
#ifdef FLASH_CAUSAL
        cb_wait_front(cb_m, 1);
#endif
    }

    // One job: its rows against key tiles [0, nkj) in blocks of KB from the lane, the
    // rows paired in steps, then each row finalized. False: a bounded wait gave up.
    bool job(
        const km::Compute& cp,
        uint32_t rows,
        uint32_t row0,
        uint32_t nkj,
        uint32_t stride,
        uint32_t& blocks
    ) {
        if (!cp.wait_front(cb_q, R * DT)) {
            return false;
        }
        uint32_t c_base = 0;
        for (uint32_t k_start = 0; k_start < nkj; k_start += KB) {
            const uint32_t kb = nkj - k_start < KB ? nkj - k_start : KB;
            const uint32_t nsub = (kb + SC - 1) / SC;
            if (!km::lane_take(cp, cb_kv, P)) {
                return false;
            }
            // Visible steps in (chunk, row) order; pairs hold two different rows.
            Step pending{};
            bool has_pending = false;
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                const uint32_t k0 = sub * SC;
                const uint32_t kg = k_start + k0;
                const uint32_t nt_block = kb - k0 < SC ? kb - k0 : SC;
                for (uint32_t r = 0; r < rows; ++r) {
#ifdef FLASH_CAUSAL
                    const uint32_t rowg = row0 + r * stride;
                    const uint32_t last_key = rowg < nkj - 1 ? rowg : nkj - 1;
#else
                    const uint32_t last_key = nkj - 1;
#endif
                    if (kg > last_key) {
                        continue;
                    }
                    const uint32_t nt =
                        (last_key - kg + 1 < nt_block ? last_key - kg + 1 : nt_block);
                    const bool last = kg + nt - 1 == last_key;
#ifdef FLASH_CAUSAL
                    const bool diag = last && last_key == rowg;
#else
                    const bool diag = false;
#endif
                    const Step s{r, c_base + sub, k0, nt, last, diag};
                    if (has_pending && pending.r != s.r) {
                        score(pending);
                        score(s);
                        output(pending);
                        output(s);
                        has_pending = false;
                    } else {
                        if (has_pending) {
                            score(pending);
                            output(pending);
                        }
                        pending = s;
                        has_pending = true;
                    }
                }
            }
            if (has_pending) {
                score(pending);
                output(pending);
            }
            km::lane_done(cb_kv, P);
            ++blocks;
            c_base += nsub;
        }
        for (uint32_t r = 0; r < rows; ++r) {
            finalize(cp, r);
        }
        cb_pop_front(cb_q, R * DT);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Compute cp;
    const uint32_t window = cp.arg(kArgWindow);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_kv, cb_p);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    MATH((flash::init(__builtin_bit_cast(float, A_BITS))));
    MATH((flash::macro_init()));
    PACK((flash::addr_init()));
    constant_tiles();
    // Row state and O are slots written in place, reserved once and never pushed.
    cb_reserve_back(cb_st, 2 * R);
    cb_reserve_back(cb_o, R * W);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegAttend || !cp.wait_front(kCbJobs, 1)) {
            return;
        }
        const uint32_t table = get_tile_address(kCbJobs, 0);
        invalidate_l1_cache();
        const uint32_t jobs = km::l1(table)[kTabJobs];
        uint32_t blocks = 0;
        bool ok = jobs <= kJobsMax;
        for (uint32_t j = 0; ok && j < jobs; ++j) {
            const volatile tt_l1_ptr uint32_t* w = (
                km::l1(table + 4 * (kTabFirst + kJobWords * j))
            );
            ok = job(cp, w[kJobRows], w[kJobStart], w[kJobKeys], w[kJobStride], blocks);
        }
        if (ok) {
            km::lane_end(cp, cb_kv, P, window, blocks);
        }
        cb_pop_front(kCbJobs, 1);
    });
    PACK((flash::restore_constants()));
}
