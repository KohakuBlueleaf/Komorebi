// Leader compute: in CARRY, adds the tiles that arrive with the leader's casts into
// one fp32 tile (the core's checksum of them).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_broadcast.hpp"
#include "leader.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t included = compute.arg(0);
    const uint32_t window = compute.arg(1);
    compute_kernel_hw_startup(kCbTiles, kCbSum);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCarry || included == 0) {
            return;
        }
        const uint32_t casts = command.a1;
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        bool ok = true;
        for (uint32_t cast = 0; ok && cast < casts; cast++) {
            ok = km::broadcast_receive(compute, kCbTiles, 1);
            if (ok) {
                copy_init(kCbTiles);
                copy_tile(kCbTiles, 0, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                km::broadcast_done(kCbTiles, 1);
            }
        }
        ok = ok && km::broadcast_end(compute, kCbTiles, 1, window, casts);
        tile_regs_commit();
        tile_regs_wait();
        if (ok && compute.reserve_back(kCbSum, 1)) {
            pack_tile(0, kCbSum);
            cb_push_back(kCbSum, 1);
        }
        tile_regs_release();
    });
}
