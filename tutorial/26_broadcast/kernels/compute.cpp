// The compute kernel: adds the blocks that arrive into one tile.
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_broadcast.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t window = compute.arg(0);
    compute_kernel_hw_startup(kRowBuffer, kOutput);
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerBroadcast) {
            return;
        }
        const uint32_t block_count = command.a1;
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        bool ok = true;
        for (uint32_t block = 0; ok && block < block_count; block++) {
            ok = km::broadcast_receive(compute, kRowBuffer, 1);
            if (ok) {
                copy_init(kRowBuffer);
                copy_tile(kRowBuffer, 0, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                km::broadcast_done(kRowBuffer, 1);
            }
        }
        ok = ok && km::broadcast_end(compute, kRowBuffer, 1, window, block_count);
        tile_regs_commit();
        tile_regs_wait();
        if (ok && compute.reserve_back(kOutput, 1)) {
            pack_tile(0, kOutput);
            cb_push_back(kOutput, 1);
        }
        tile_regs_release();
    });
    // --8<-- [end:run]
}
