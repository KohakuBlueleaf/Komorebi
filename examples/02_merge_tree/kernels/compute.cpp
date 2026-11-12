// Merge-tree compute: adds input tile t into sum t % P (fp32, P tiles in DST), adds the
// parts it receives in rank order, and packs the P tiles: its part or the total.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_port.hpp"
#include "merge.h"

namespace {

    // DST d += tile `page` of buffer cb (copied to DST kPartMax, added by the SFPU).
    void add_page(uint32_t cb, uint32_t page, uint32_t d) {
        copy_init(cb);
        copy_tile(cb, page, kPartMax);
        add_binary_tile_init();
        add_binary_tile(d, kPartMax, d);
    }

}  // namespace

void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMerge) {
            return;
        }
        const km::Group g = km::team_group(r, cp.me());
        const km::Share s = km::share(r.a1, g.rank, g.size);
        const km::Arrival arrival = {kCbArrive, kArrivePages, r.a7};
        tile_regs_acquire();
        fill_tile_init();
        for (uint32_t d = 0; d < r.a7; d++) {
            fill_tile(d, 0.0f);
        }
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t i = 0; ok && i < s.len; i++) {
            ok = cp.wait_front(kCbIn, 1);
            if (ok) {
                add_page(kCbIn, 0, (s.start + i) % r.a7);
                cb_pop_front(kCbIn, 1);
            }
        }
        ok = ok && km::reduce(cp, arrival, g, r.a4, [&](uint32_t k) {
                 for (uint32_t d = 0; d < r.a7; d++) {
                     add_page(kCbArrive, km::slot_page(arrival, k) + d, d);
                 }
             });
        tile_regs_commit();
        tile_regs_wait();
        if (ok && cp.reserve_back(kCbOut, r.a7)) {  // a give-up: no output tiles
            for (uint32_t d = 0; d < r.a7; d++) {
                pack_tile(d, kCbOut);
            }
            cb_push_back(kCbOut, r.a7);
        }
        tile_regs_release();
    });
}
