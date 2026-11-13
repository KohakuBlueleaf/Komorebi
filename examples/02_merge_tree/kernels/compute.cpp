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

    // DST 0 to n - 1 = 0: the part's sums.
    void clear_sums(uint32_t n) {
        fill_tile_init();
        for (uint32_t d = 0; d < n; d++) {
            fill_tile(d, 0.0f);
        }
    }

}  // namespace

void kernel_main() {
    km::Compute cp;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    // The runtime's loop: the body runs once per command whose group holds this core.
    cp.run([&](const km_command_t& r) {
        if (km::handler(r) !=
            kHandlerMerge) {  // this program answers MERGE commands only
            return;
        }
        const uint32_t tiles = r.a1;  // input tiles T of the job
        const uint32_t form = r.a4;   // 0: merge at one owner, 1: along a tree
        const uint32_t part = r.a7;   // tiles per part P (1 to 3), one sum each in DST
        // The group is the command's range of cores; my rank in it gives my partition
        // of T.
        const km::Group g = km::group_of(r, cp.me());
        const km::Partition s = km::partition(tiles, g.rank, g.size);
        const km::Arrival arrival = {kCbArrive, kArrivePages, part};
        tile_regs_acquire();  // DST, the vector unit's registers, until the release
        clear_sums(part);     // the P sums start at zero
        // My own tiles, as the reader brings them from DRAM: tile t into sum t mod P.
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t i = 0; ok && i < s.len; i++) {
            ok = cp.wait_front(kCbIn, 1);
            if (ok) {
                add_page(kCbIn, 0, (s.start + i) % part);
                cb_pop_front(kCbIn, 1);
            }
        }
        // The reduce: wait for the parts other cores of the group send me, add them in
        // rank order (by owner: rank 0 takes all; by tree: one partner a round).
        ok = ok && km::reduce(cp, arrival, g, form, [&](uint32_t k) {
                 for (uint32_t d = 0; d < part; d++) {
                     add_page(kCbArrive, km::slot_page(arrival, k) + d, d);
                 }
             });
        tile_regs_commit();
        tile_regs_wait();
        // The P sums to the writer, which sends them on or (rank 0) writes the total.
        // After a give-up (a wait ran out, or the run was stopped) nothing is packed.
        if (ok && cp.reserve_back(kCbOut, part)) {
            for (uint32_t d = 0; d < part; d++) {
                pack_tile(d, kCbOut);
            }
            cb_push_back(kCbOut, part);
        }
        tile_regs_release();
    });
}
