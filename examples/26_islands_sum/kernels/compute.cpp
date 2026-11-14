// Island-aware sum, compute: input tile t into sum t % P (fp32, in DST), its island's
// parts (km::local), at the first leader the other leaders' parts; packs the P tiles.
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
#include "sum.h"

namespace {

    // DST sum_index += tile `page` of buffer cb (copied to DST kPartMax, added by the
    // SFPU).
    void add_page(uint32_t cb, uint32_t page, uint32_t sum_index) {
        copy_init(cb);
        copy_tile(cb, page, kPartMax);
        add_binary_tile_init();
        add_binary_tile(sum_index, kPartMax, sum_index);
    }

    // DST 0 to sums - 1 = 0: the part's sums.
    void clear_sums(uint32_t sums) {
        fill_tile_init();
        for (uint32_t sum_index = 0; sum_index < sums; sum_index++) {
            fill_tile(sum_index, 0.0f);
        }
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    // The runtime's loop: the body runs once per command whose group holds this core.
    compute.run([&](const km_command_t& command) {
        if (
            km::handler(command) != kHandlerSum
        ) {  // this program answers MERGE commands only
            return;
        }
        const uint32_t tiles = command.a1;  // input tiles T of the job
        const uint32_t form = command.a4;   // 0: merge at one owner, 1: along a tree
        const uint32_t part_tiles = command
                                        .a7;  // tiles per part P (1 to 3): a sum each
        // The group is the command's range of cores; my rank in it gives my partition
        // of T.
        const km::Group group = km::group_of(command, compute.me());
        const km::Partition partition = km::partition(tiles, group.rank, group.size);
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        tile_regs_acquire();     // DST, the vector unit's registers, until the release
        clear_sums(part_tiles);  // the P sums start at zero
        // My own tiles, as the reader brings them from DRAM: tile t into sum t mod P.
        bool ok = true;  // the same on all three threads (wait_front tells them)
        for (uint32_t tile = 0; ok && tile < partition.len; tile++) {
            ok = compute.wait_front(kCbIn, 1);
            if (ok) {
                add_page(kCbIn, 0, (partition.start + tile) % part_tiles);
                cb_pop_front(kCbIn, 1);
            }
        }
        const auto combine = [&](uint32_t slot) {
            for (uint32_t sum_index = 0; sum_index < part_tiles; sum_index++) {
                add_page(
                    kCbArrive, km::slot_page(arrival, slot) + sum_index, sum_index
                );
            }
        };
        // The reduce inside my island, in rank order; then, at the first leader, one
        // part from each other island's leader, in island order.
        ok = ok && km::reduce(compute, arrival, group, form, km::local, combine);
        const km::Group leaders = group.leaders();
        if (ok && leaders.rank == 0 && leaders.size > 1) {
            ok = km::take(compute, arrival);
            for (uint32_t j = 1; ok && j < leaders.size; j++) {
                combine(cross_slot(part_tiles, j));
            }
            if (ok) {
                km::untake(arrival);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        // The P sums to the writer, which sends them on or (rank 0) writes the total.
        // After a give-up (a wait ran out, or the run was stopped) nothing is packed.
        if (ok && compute.reserve_back(kCbOut, part_tiles)) {
            for (uint32_t sum_index = 0; sum_index < part_tiles; sum_index++) {
                pack_tile(sum_index, kCbOut);
            }
            cb_push_back(kCbOut, part_tiles);
        }
        tile_regs_release();
    });
}
