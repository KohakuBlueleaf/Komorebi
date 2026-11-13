// Map/collect compute: MAP adds the core's input tiles into one tile (a zero tile
// when its partition is empty).
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
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMap) {
            return;
        }
        const uint32_t tiles = command.a1;
        const uint32_t slow_cycles = command.a4;
        const uint32_t slow_rank = command.a5;
        const km::Partition partition = (
            km::partition(tiles, compute.me() - km::first(command), km::count(command))
        );
        tile_regs_acquire();
        bool ok = true;  // the same on all three threads (wait_front tells them)
        if (partition.len == 0) {
            fill_tile_init();
            fill_tile(0, 0.0f);
        } else {
            // The sum stays in DST (fp32): each tile is copied to DST 1, the SFPU adds
            // it.
            copy_init(kCbIn);
            ok = compute.wait_front(kCbIn, 1);
            if (ok) {
                copy_tile(kCbIn, 0, 0);
                cb_pop_front(kCbIn, 1);
            }
            for (uint32_t tile = 1; ok && tile < partition.len; tile++) {
                ok = compute.wait_front(kCbIn, 1);
                if (ok) {
                    copy_init(kCbIn);
                    copy_tile(kCbIn, 0, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                    cb_pop_front(kCbIn, 1);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        if (
            compute.me() - km::first(command) == slow_rank
        ) {  // slow compute (test only)
            PACK(({
                const uint64_t start = km::wall64();
                while (km::wall64() - start < slow_cycles) {
                    ;
                }
            }));
        }
        if (ok && compute.reserve_back(kCbOut, 1)) {  // a give-up: no output tile
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
    });
}
