// Broadcast compute: adds every tile of the core's row block and column block, for
// each block, into one fp32 tile (its checksum).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "broadcast.h"
#include "komorebi/device/movement/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    const bool included = compute.arg(0) != 0;
    const uint32_t window = compute.arg(1), pages = compute.arg(2);
    compute_kernel_hw_startup(kCbRow, kCbOut);
    compute.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if ((handler != kHandlerBroadcast && handler != kHandlerRead) || !included) {
            return;
        }
        const uint32_t blocks = command.a3;
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t block = 0; ok && block < blocks; block++) {
            for (const uint32_t cb : {kCbRow, kCbCol}) {
                ok = km::broadcast_receive(compute, cb, pages);
                if (!ok) {
                    break;
                }
                for (uint32_t page = 0; page < pages; page++) {  // a page of the block,
                                                                 // by index
                    copy_init(cb);
                    copy_tile(cb, page, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                km::broadcast_done(cb, pages);
            }
        }
        ok = (
            ok && km::broadcast_end(compute, kCbRow, pages, window, blocks) &&
            km::broadcast_end(compute, kCbCol, pages, window, blocks)
        );
        tile_regs_commit();
        tile_regs_wait();
        if (ok && compute.reserve_back(kCbOut, 1)) {
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
    });
}
