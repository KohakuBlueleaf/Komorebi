// Tree experiment, compute: the rank's P tiles into DST, the arrivals added in rank
// order, packed; in an all-reduce a rank other than 0 then packs the sum cast to it.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/movement/km_tree.hpp"
#include "tree.h"

namespace {

    // DST sum_index += tile `page` of buffer cb (copied to DST kPartMax, added by the
    // SFPU).
    void add_page(uint32_t cb, uint32_t page, uint32_t sum_index) {
        copy_init(cb);
        copy_tile(cb, page, kPartMax);
        add_binary_tile_init();
        add_binary_tile(sum_index, kPartMax, sum_index);
    }

    // The P tiles of DST to kCbOut. False: stopping.
    bool pack_out(const km::Compute& compute, uint32_t part_tiles) {
        if (!compute.reserve_back(kCbOut, part_tiles)) {
            return false;
        }
        for (uint32_t sum_index = 0; sum_index < part_tiles; sum_index++) {
            pack_tile(sum_index, kCbOut);
        }
        cb_push_back(kCbOut, part_tiles);
        return true;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbIn, kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerAllReduce) {
            return;
        }
        const uint32_t form = command.a2;
        const uint32_t part_tiles = command.a3;
        const km::Group group = km::group_of(command, compute.me());
        const km::Arrival arrival = {kCbArrive, kArrivePages, part_tiles};
        tile_regs_acquire();
        bool ok = compute.wait_front(kCbIn, part_tiles);
        if (ok) {
            copy_init(kCbIn);
            for (uint32_t t = 0; t < part_tiles; t++) {
                copy_tile(kCbIn, t, t);
            }
            cb_pop_front(kCbIn, part_tiles);
        }
        ok = ok && km::reduce(compute, arrival, group, form, [&](uint32_t slot) {
            for (uint32_t sum_index = 0; sum_index < part_tiles; sum_index++) {
                add_page(
                    kCbArrive, km::slot_page(arrival, slot) + sum_index, sum_index
                );
            }
        });
        tile_regs_commit();
        tile_regs_wait();
        ok = ok && pack_out(compute, part_tiles);
        tile_regs_release();
        if (!ok || command.a4 == 0) {
            return;
        }
        // the sum cast back: slot 0's P tiles, packed for the writer
        km::all_reduce_take(compute, arrival, group, [&] {
            tile_regs_acquire();
            copy_init(kCbArrive);
            for (uint32_t t = 0; t < part_tiles; t++) {
                copy_tile(kCbArrive, t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_out(compute, part_tiles);
            tile_regs_release();
        });
    });
}
