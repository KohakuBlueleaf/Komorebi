// The compute kernel: multiplies the two input tiles as matrices and puts the product
// in the output buffer.
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kFirstInput, kSecondInput, kOutput);
    matmul_init(kFirstInput, kSecondInput);
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMultiply) {
            return;
        }
        if (!compute.wait_front(kFirstInput, 1) ||
            !compute.wait_front(kSecondInput, 1)) {
            return;
        }
        tile_regs_acquire();
        matmul_tiles(kFirstInput, kSecondInput, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (compute.reserve_back(kOutput, 1)) {
            pack_tile(0, kOutput);
            cb_push_back(kOutput, 1);
        }
        tile_regs_release();
        cb_pop_front(kFirstInput, 1);
        cb_pop_front(kSecondInput, 1);
    });
    // --8<-- [end:run]
}
