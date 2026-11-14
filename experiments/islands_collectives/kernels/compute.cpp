// Collectives across islands, compute: a7 rounds of kBusyTiles matrix products (no
// pack) beside GATHER and REDUCE.
#include <stdint.h>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "collectives.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kCbA, kCbB, kCbOut);
    matmul_init(kCbA, kCbB);
    compute.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerGather && handler != kHandlerReduce) {
            return;
        }
        // the operands' tiles as their buffers hold them (only the unit's load counts)
        for (uint32_t round = 0; round < command.a7; round++) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < kBusyTiles; t++) {
                matmul_tiles(kCbA, kCbB, 0, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
    });
}
