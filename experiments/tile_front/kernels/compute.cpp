// SiLU over a stream of fp32 tiles on one core: BODY 0 the hand body (tt-metal's
// silu_tile, as the dense project's SwiGLU layer calls it), BODY 1 the emitted one.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "front.hpp"
#if BODY == 1
#include "silu.hpp"
#endif

namespace {

    inline void body_init() {
#if BODY == 1
        MATH((km_tile::silu_init()));
#else
        silu_tile_init();
#endif
    }

    inline void body(uint32_t dst) {
#if BODY == 1
        MATH((km_tile::silu(dst)));
#else
        silu_tile(dst);
#endif
    }

}  // namespace

void kernel_main() {
    const uint32_t nonce = get_arg_val<uint32_t>(0);
    const uint32_t tiles = get_arg_val<uint32_t>(1);
    volatile uint32_t* info = (
        reinterpret_cast<volatile uint32_t*>(get_tile_address(front::kCbInfo, 0))
    );
    compute_kernel_hw_startup(front::kCbIn, front::kCbOut);
    copy_init(front::kCbIn);
    body_init();
    // The empty span: a wall-clock read pair around one tensix_sync, on the math CPU.
    for (uint32_t i = 0; i < front::kSyncs; ++i) {
        MATH(
            const uint32_t start = front::wall(); tensix_sync();
            info[front::kSync + i] = front::wall() - start
        );
    }
    for (uint32_t t = 0; t < tiles && t < front::kMaxTiles; ++t) {
        const uint32_t started = front::wall();
        cb_wait_front(front::kCbIn, 1);
        tile_regs_acquire();
        copy_tile(front::kCbIn, 0, 0);
        MATH(tensix_sync(); const uint32_t start = front::wall());
        body(0);
        MATH(tensix_sync(); info[front::kBody + t] = front::wall() - start);
        tile_regs_commit();
        cb_pop_front(front::kCbIn, 1);
        tile_regs_wait();
        cb_reserve_back(front::kCbOut, 1);
        pack_tile(0, front::kCbOut);
        tile_regs_release();
        cb_push_back(front::kCbOut, 1);
        PACK(info[front::kLoop + t] = front::wall() - started);
    }
    MATH(asm volatile("fence" :: : "memory"); info[front::kMathDone] = nonce);
#ifdef TRISC_PACK
    uint32_t error = 0, polls = 0;
    while (true) {
        asm volatile("fence" ::: "memory");
        if (info[front::kMathDone] == nonce) {
            break;
        }
        if (++polls >= front::kSpinLimit) {
            error = front::kErrorSpin;
            break;
        }
    }
    info[front::kTiles] = tiles;
    info[front::kError] = error;
    asm volatile("fence" ::: "memory");
    info[front::kNonce] = nonce;
    asm volatile("fence" ::: "memory");
    cb_reserve_back(front::kCbInfo, 1);
    cb_push_back(front::kCbInfo, 1);
#endif
}
