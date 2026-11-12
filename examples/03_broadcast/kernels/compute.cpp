// Broadcast compute: adds every tile of the member's row block and column block, for
// each block k, into one fp32 tile (its checksum).
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
#include "komorebi/device/km_lane.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute cp;
    const bool member = cp.arg(0) != 0;
    const uint32_t window = cp.arg(1), pages = cp.arg(2);
    compute_kernel_hw_startup(kCbRow, kCbOut);
    cp.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if ((seg != kSegCast && seg != kSegRead) || !member) {
            return;
        }
        const uint32_t n = r.a3;
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t k = 0; ok && k < n; k++) {
            for (const uint32_t cb : {kCbRow, kCbCol}) {
                ok = km::lane_take(cp, cb, pages);
                if (!ok) {
                    break;
                }
                for (uint32_t i = 0; i < pages; i++) {  // page i of block k, by index
                    copy_init(cb);
                    copy_tile(cb, i, 1);
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                km::lane_done(cb, pages);
            }
        }
        ok =
            (ok && km::lane_end(cp, kCbRow, pages, window, n) &&
             km::lane_end(cp, kCbCol, pages, window, n));
        tile_regs_commit();
        tile_regs_wait();
        if (ok && cp.reserve_back(kCbOut, 1)) {
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
    });
}
