// The compute kernel: squares each tile of the core's share and adds the squares into
// one tile, then adds the sums of the other cores to it.
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/movement/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kOwnTiles, kOutput);
    const km::Arrival arrival = {kArrival, kArrivalPages, 1};
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerSumSquares) {
            return;
        }
        const km::Group group = km::group_of(command, compute.me());
        const km::Partition share = km::partition(command.a5, group.rank, group.size);
        tile_regs_acquire();
        bool ok = true;
        for (uint32_t index = 0; ok && index < share.len; index++) {
            ok = compute.wait_front(kOwnTiles, 1);
            if (ok) {
                const uint32_t into = index == 0 ? 0 : 1;
                copy_init(kOwnTiles);
                copy_tile(kOwnTiles, 0, into);
                mul_binary_tile_init();
                mul_binary_tile(into, into, into);
                if (index > 0) {
                    add_binary_tile_init();
                    add_binary_tile(0, 1, 0);
                }
                cb_pop_front(kOwnTiles, 1);
            }
        }
        ok = ok && km::reduce(compute, arrival, group, 0, [&](uint32_t slot) {
            copy_init(kArrival);
            copy_tile(kArrival, km::slot_page(arrival, slot), 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        });
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
