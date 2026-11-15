// The epilogue program's compute: per tile its two operands copied into Dst 0 and 1,
// the emitted body (generated/epilogue.hpp) timed on the math CPU, Dst 0 packed.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "epilogue.hpp"
#include "layout.hpp"

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(layout::kCbInfo, 0))
    );
    compute_kernel_hw_startup(layout::kCbA, layout::kCbOut);
    copy_init(layout::kCbA);
    MATH((km_tile::epilogue_init()));
    // The empty span: a wall-clock read pair around one tensix_sync, on the math CPU.
    for (uint32_t i = 0; i < layout::kSyncs; ++i) {
        MATH(
            const uint32_t start = layout::wall(); tensix_sync();
            info[layout::kSync + i] = layout::wall() - start
        );
    }
    MATH(const uint32_t first = layout::wall());
    for (uint32_t t = 0; t < tiles; ++t) {
        cb_wait_front(layout::kCbA, 1);
        cb_wait_front(layout::kCbB, 1);
        tile_regs_acquire();
        copy_tile(layout::kCbA, 0, 0);
        copy_tile(layout::kCbB, 0, 1);
        MATH(tensix_sync(); const uint32_t start = layout::wall());
        MATH((km_tile::epilogue(0)));
        MATH(tensix_sync(); const uint32_t span = layout::wall() - start);
        MATH(if (t < layout::kBodies) { info[layout::kBody + t] = span; });
        tile_regs_commit();
        cb_pop_front(layout::kCbA, 1);
        cb_pop_front(layout::kCbB, 1);
        tile_regs_wait();
        cb_reserve_back(layout::kCbOut, 1);
        pack_tile(0, layout::kCbOut);
        tile_regs_release();
        cb_push_back(layout::kCbOut, 1);
    }
    MATH(info[layout::kStream] = layout::wall() - first);
    MATH(asm volatile("fence" :: : "memory"); info[layout::kMathDone] = nonce);
#ifdef TRISC_PACK
    uint32_t error = 0, polls = 0;
    while (true) {
        asm volatile("fence" ::: "memory");
        if (info[layout::kMathDone] == nonce) {
            break;
        }
        if (++polls >= layout::kSpinLimit) {
            error = layout::kErrorSpin;
            break;
        }
    }
    info[layout::kTiles] = tiles;
    info[layout::kError] = error;
    asm volatile("fence" ::: "memory");
    info[layout::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(layout::kCbInfo, 1);
    cb_push_back(layout::kCbInfo, 1);
#endif
}
