// Tile-rules product: the dense project's compute order (unpack, math, pack) for one
// tile of A times one of B a unit, by the library's matmul_block_no_mop (BODY 22) or
// by its unpack and the words komorebi.dsl.tile emits for its math (BODY 23). The
// product is timed on the math CPU as the attention body times its score products:
// from its issue (no sync before it: the unpacker is ahead holding both Src banks) to
// a tensix_sync after it; the empty span (kSync) is taken off on the host.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/matmul_custom.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "rules.hpp"
#if BODY == 23
#include "product_hifi2.hpp"
#endif

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(rules::kCbInfo, 0))
    );
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        rules::kCbA, rules::kCbB, rules::kCbOut
    );
    mm_no_mop_init_short(rules::kCbA, rules::kCbB, false, 1, 1, 1);
    // The empty span: a wall-clock read pair around one tensix_sync, on the math CPU.
    for (uint32_t i = 0; i < rules::kSyncs; ++i) {
        MATH(
            const uint32_t start = rules::wall(); tensix_sync();
            info[rules::kSync + i] = rules::wall() - start
        );
    }
    for (uint32_t u = 0; u < units && u < rules::kMaxUnits; ++u) {
        const uint32_t started = rules::wall();
        cb_wait_front(rules::kCbA, 1);
        cb_wait_front(rules::kCbB, 1);
        tile_regs_acquire();
        MATH(const uint32_t start = rules::wall());
#if BODY == 22
        matmul_block_no_mop(rules::kCbA, rules::kCbB, 0, 0, 0, false, 1, 1, 1);
#else
        UNPACK((llk_unpack_AB_matmul(rules::kCbA, rules::kCbB, 0, 0, 1, 1, 1)));
        MATH((km_tile::product_hifi2(0)));
#endif
        MATH(tensix_sync(); info[rules::kBody + u] = rules::wall() - start);
        tile_regs_commit();
        cb_pop_front(rules::kCbA, 1);
        cb_pop_front(rules::kCbB, 1);
        tile_regs_wait();
        cb_reserve_back(rules::kCbOut, 1);
        pack_tile(0, rules::kCbOut);
        tile_regs_release();
        cb_push_back(rules::kCbOut, 1);
        PACK(info[rules::kLoop + u] = rules::wall() - started);
    }
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
    info[rules::kUnits] = units;
    info[rules::kError] = error;
    asm volatile("fence" ::: "memory");
    info[rules::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(rules::kCbInfo, 1);
    cb_push_back(rules::kCbInfo, 1);
#endif
}
