// Formats probe compute: CASE 0 the left tile times right tiles first.., CASE 1 a
// copy through Dst, CASE 2 the product with the right tile read transposed, then Dst's.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "layout.hpp"

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(formats::kCbInfo, 0))
    );
#if CASE == 1
    compute_kernel_hw_startup(formats::kCbA, formats::kCbA, formats::kCbOut);
    copy_init(formats::kCbA);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(
        formats::kCbA, formats::kCbB, formats::kCbOut
    );
    matmul_init(formats::kCbA, formats::kCbB, CASE == 2 ? 1 : 0);
#endif
    for (uint32_t i = 0; i < formats::kSyncs; ++i) {
        MATH(
            const uint32_t start = formats::wall(); tensix_sync();
            info[formats::kSync + i] = formats::wall() - start
        );
    }
    cb_wait_front(formats::kCbA, 1);
#if CASE != 1
    cb_wait_front(formats::kCbB, first + count);
#endif
    for (uint32_t t = 0; t < count && t < formats::kMaxTiles; ++t) {
        tile_regs_acquire();
        MATH(const uint32_t start = formats::wall());
#if CASE == 1
        copy_tile(formats::kCbA, 0, 0);
#else
        matmul_tiles(formats::kCbA, formats::kCbB, 0, first + t, 0);
#endif
        MATH(tensix_sync(); info[formats::kProduct + t] = formats::wall() - start);
#if CASE == 2
        transpose_dest_init<true, true>();
        MATH(const uint32_t turned = formats::wall());
        transpose_dest<true, true>(0);
        MATH(tensix_sync(); info[formats::kTranspose + t] = formats::wall() - turned);
#endif
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(formats::kCbOut, 1);
        pack_tile(0, formats::kCbOut);
        cb_push_back(formats::kCbOut, 1);
        tile_regs_release();
    }
    MATH(asm volatile("fence" :: : "memory"); info[formats::kMathDone] = nonce);
#ifdef TRISC_PACK
    uint32_t error = 0, polls = 0;
    while (true) {
        asm volatile("fence" ::: "memory");
        if (info[formats::kMathDone] == nonce) {
            break;
        }
        if (++polls >= formats::kSpinLimit) {
            error = formats::kErrorSpin;
            break;
        }
    }
    info[formats::kCount] = count;
    info[formats::kError] = error;
    asm volatile("fence" ::: "memory");
    info[formats::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(formats::kCbInfo, 1);
    cb_push_back(formats::kCbInfo, 1);
#endif
}
