// The compute kernel: starts from the core's own tile and adds each tile that arrives.
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/km_port.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kOwnTile, kOutput);
    const km::Arrival arrival = {kArrival, kArrivalPages, 1};
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerReduce) {
            return;
        }
        const km::Group group = km::group_of(command, compute.me());
        const uint32_t form = command.a4;
        tile_regs_acquire();
        bool ok = compute.wait_front(kOwnTile, 1);
        if (ok) {
            copy_init(kOwnTile);
            copy_tile(kOwnTile, 0, 0);
            cb_pop_front(kOwnTile, 1);
        }
        ok = ok && km::reduce(compute, arrival, group, form, [&](uint32_t slot) {
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
