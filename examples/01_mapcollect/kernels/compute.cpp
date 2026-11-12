// Map/collect compute: MAP adds the member's input tiles into one tile (a zero tile
// when its share is empty).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_worker.hpp"
#include "mapcollect.h"

void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMap) {
            return;
        }
        const km::Share s = km::share(r.a1, cp.me() - km::first(r), km::count(r));
        tile_regs_acquire();
        if (s.len == 0) {
            fill_tile_init();
            fill_tile(0, 0.0f);
        } else {
            // The sum stays in DST (fp32): each tile is copied to DST 1, the SFPU adds
            // it.
            copy_init(kCbIn);
            cb_wait_front(kCbIn, 1);
            copy_tile(kCbIn, 0, 0);
            cb_pop_front(kCbIn, 1);
            for (uint32_t i = 1; i < s.len; i++) {
                cb_wait_front(kCbIn, 1);
                copy_init(kCbIn);
                copy_tile(kCbIn, 0, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                cb_pop_front(kCbIn, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(kCbOut, 1);
        pack_tile(0, kCbOut);
        cb_push_back(kCbOut, 1);
        tile_regs_release();
    });
}
