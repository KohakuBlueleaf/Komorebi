// Ceilings compute: MATMUL's passes over the resident tiles (READ: nothing).
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reg_api.h"
#include "ceilings.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup<SrcOrder::Reverse>(kCbA, kCbB, kCbOut);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMatmul) {
            return;
        }
        matmul_block_init(kCbA, kCbB, 0, kSw, kSh, kKb);
        for (uint32_t pass = 0; pass < command.a0; ++pass) {
            for (uint32_t row = 0; row < kBm; row += kSh) {
                for (uint32_t column = 0; column < kBn; column += kSw) {
                    tile_regs_acquire();
                    for (uint32_t k = 0; k < kKb; ++k) {
                        matmul_block(
                            kCbA, kCbB, row * kKb + k, k * kBn + column, 0, 0, kSw, kSh,
                            kKb
                        );
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    tile_regs_release();
                }
            }
        }
    });
}
