// The per-core attention body, one query row: score products, the softmax step (BODY 18
// hand, 19 emitted), P packed (bf16), P V added into O (fp32) by L1 accumulation.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "rules.hpp"
#if BODY == 19
#include "softstep.hpp"
#endif
#if BODY == 18 && defined(TRISC_MATH)
#include "sfpu.hpp"  // projects/attention/kernels
#endif

namespace {

    constexpr uint32_t kHeadTiles = HEAD_TILES, kValueTiles = VALUE_TILES;
    constexpr uint32_t kKeys = 3;  // key tiles a step
    constexpr uint32_t kFirstValue = kHeadTiles * kKeys;
    constexpr uint32_t kBlockPages = kFirstValue + kKeys * kValueTiles;
    // O's columns, then (the hand body) the row sum l from the ones tile.
    constexpr uint32_t kColumns = kValueTiles + (BODY == 18 ? 1 : 0);

    // Packer output format: fp32 (state, O) or bf16 (P), changed on transitions only
    // (projects/attention's pack_format).
    bool pack_fp32 = false;

    inline void pack_format(bool fp32) {
        if (fp32 != pack_fp32) {
            pack_reconfig_data_format(
                fp32 ? rules::kCbProb : rules::kCbOut,
                fp32 ? rules::kCbOut : rules::kCbProb
            );
            pack_fp32 = fp32;
        }
    }

    inline void vector_step(bool first) {
#if BODY == 18
        MATH(
            (
                ckernel::math::set_dst_write_addr<
                    DstTileShape::Tile32x32,
                    UnpackDestination::SrcRegs>(0)
            )
        );
        MATH((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH)));
        MATH((flash::state_max<3>()));
        if (first) {
            MATH((flash::state_update<true>(STEP_THRESHOLD)));
        } else {
            MATH((flash::state_update<false>(STEP_THRESHOLD)));
        }
        MATH((flash::exp_step<3>()));
#else
        (void)first;
        MATH((km_tile::softstep(0)));
#endif
    }

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t steps = get_arg_val<uint32_t>(1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(rules::kCbInfo, 0))
    );
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        rules::kCbQ, rules::kCbKv, rules::kCbProb
    );
#if BODY == 18
    MATH((flash::init(EXP_A)));
    MATH((flash::macro_init()));
    MATH((flash::addr_init()));
#else
    MATH((km_tile::softstep_init()));
#endif
    cb_wait_front(rules::kCbQ, kHeadTiles);
    cb_wait_front(rules::kCbOnes, 1);
    PACK((cb_reserve_back(rules::kCbOut, kColumns + 1)));
    for (uint32_t s = 0; s < steps && s < rules::kMaxUnits; ++s) {
        const uint32_t started = rules::wall();
        const bool first = s == 0, last = s + 1 == steps;
        cb_wait_front(rules::kCbKv, kBlockPages);
        cb_wait_front(rules::kCbSt, 1);
        // Score: S = Q K^T for the step's 3 key tiles, the state beside it.
        tile_regs_acquire_math_clear();
        reconfig_data_format_srca(rules::kCbKv, rules::kCbSt);
        copy_init(rules::kCbSt);
        copy_tile(rules::kCbSt, 0, 3);
        reconfig_data_format_srca(rules::kCbSt, rules::kCbKv);
        mm_no_mop_init_short(rules::kCbQ, rules::kCbKv, true, kKeys, 1, kHeadTiles);
        // No sync before a product: the unpacker is ahead holding both Src banks.
        MATH(const uint32_t t0 = rules::wall());
        for (uint32_t h = 0; h < kHeadTiles; ++h) {
            matmul_block_no_mop(
                rules::kCbQ, rules::kCbKv, h, h * kKeys, 0, true, kKeys, 1, kHeadTiles
            );
        }
        MATH(tensix_sync(); const uint32_t t1 = rules::wall());
        vector_step(first);
        MATH(tensix_sync(); const uint32_t t2 = rules::wall());
        MATH(info[rules::kScore + s] = t1 - t0; info[rules::kVector + s] = t2 - t1);
        tile_regs_commit();
        cb_pop_front(rules::kCbSt, 1);
        tile_regs_wait();
        pack_format(false);
        cb_reserve_back(rules::kCbProb, kKeys);
        for (uint32_t t = 0; t < kKeys; ++t) {
            pack_tile(t, rules::kCbProb);
        }
        cb_push_back(rules::kCbProb, kKeys);
        pack_format(true);
        if (last) {
            PACK((pack_reconfig_l1_acc(0)));
            pack_tile<true>(3, rules::kCbOut, kColumns);  // the last row state
        } else {
            cb_reserve_back(rules::kCbSt, 1);
            pack_tile(3, rules::kCbSt);
            cb_push_back(rules::kCbSt, 1);
        }
        tile_regs_release_math_clear();
        // P V (and the hand body's l = P 1) added into O.
        cb_wait_front(rules::kCbProb, kKeys);
        tile_regs_acquire_math_clear();
        mm_no_mop_init_short(
            rules::kCbProb, rules::kCbKv, false, kValueTiles, 1, kKeys
        );
        MATH(info[rules::kProduct + s] = rules::wall());
        for (uint32_t key = 0; key < kKeys; ++key) {
            matmul_block_no_mop(
                rules::kCbProb,
                rules::kCbKv,
                key,
                kFirstValue + key * kValueTiles,
                0,
                false,
                kValueTiles,
                1,
                kKeys
            );
        }
#if BODY == 18
        mm_no_mop_init_short(rules::kCbProb, rules::kCbOnes, false, 1, 1, kKeys);
        for (uint32_t key = 0; key < kKeys; ++key) {
            matmul_block_no_mop(
                rules::kCbProb, rules::kCbOnes, key, 0, kValueTiles, false, 1, 1, kKeys
            );
        }
#endif
        MATH(asm volatile("fence" :: : "memory"));
        tile_regs_commit();
        tile_regs_wait();
        // The product's end on the packer: the math CPU's sync here would let the
        // unpacker run into the next step's products.
        PACK(
            asm volatile("fence" :: : "memory");
            info[rules::kProduct + s] = rules::wall() - info[rules::kProduct + s]
        );
        pack_format(true);
        PACK((pack_reconfig_l1_acc(first ? 0 : 1)));
        for (uint32_t t = 0; t < kColumns; ++t) {
            pack_tile<true>(t, rules::kCbOut, t);
        }
        PACK((pack_reconfig_l1_acc(0)));
        tile_regs_release_math_clear();
        cb_pop_front(rules::kCbProb, kKeys);
        cb_pop_front(rules::kCbKv, kBlockPages);
        MATH(info[rules::kLoop + s] = rules::wall() - started);
    }
    PACK((cb_push_back(rules::kCbOut, kColumns + 1)));
    // The empty span, after the steps: a sync while the unpacker runs ahead never
    // returns.
    for (uint32_t i = 0; i < rules::kSyncs; ++i) {
        MATH(
            const uint32_t start = rules::wall(); tensix_sync();
            info[rules::kSync + i] = rules::wall() - start
        );
    }
#if BODY == 18
    MATH((flash::restore_constants()));
#endif
    MATH(asm volatile("fence" :: : "memory"); info[rules::kMathDone] = nonce);
#ifdef TRISC_PACK
    uint32_t error = 0, polls = 0;
    while (true) {
        asm volatile("fence" ::: "memory");
        if (info[rules::kMathDone] == nonce) {
            break;
        }
        if (++polls >= rules::kSpinLimit) {
            error = rules::kErrorSpin;
            break;
        }
    }
    info[rules::kUnits] = steps;
    info[rules::kError] = error;
    asm volatile("fence" ::: "memory");
    info[rules::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(rules::kCbInfo, 1);
    cb_push_back(rules::kCbInfo, 1);
#endif
}
