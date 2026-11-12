// Merge-tree compute: adds the member's input tiles into one tile (fp32, in DST), adds
// the parts it receives in rank order, and packs the result: its part or the total.
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

    // DST 0 += tile `page` of buffer cb (copied to DST 1, added by the vector unit).
    void add_page(uint32_t cb, uint32_t page) {
        copy_init(cb);
        copy_tile(cb, page, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }

}  // namespace

void kernel_main() {
    constexpr km::Arrival arrival = {kCbArrive, kArrivePages};
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    cp.run([&](const km_record_t& r) {
        if (km::seg(r) != kSegMerge) {
            return;
        }
        const km::Group g = km::team_group(r, cp.me());
        const km::Share s = km::share(r.a1, g.rank, g.size);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t i = 0; ok && i < s.len; i++) {
            ok = cp.wait_front(kCbIn, 1);
            if (ok) {
                add_page(kCbIn, 0);
                cb_pop_front(kCbIn, 1);
            }
        }
        ok = ok && km::reduce(cp, arrival, g, r.a4, [](uint32_t k) {
                 add_page(kCbArrive, k);
             });
        tile_regs_commit();
        tile_regs_wait();
        if (ok && cp.reserve_back(kCbOut, 1)) {  // a give-up: no output tile
            pack_tile(0, kCbOut);
            cb_push_back(kCbOut, 1);
        }
        tile_regs_release();
    });
}
