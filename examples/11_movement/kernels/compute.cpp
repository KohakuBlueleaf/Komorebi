// The example's compute kernel: takes the blocks in order, each from the half the
// pattern names; READ drops them, COPY copies each into the same half of the output.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "movement.h"

constexpr km::BufferPair kInput = {kInputMoverA, kInputMoverB};
constexpr km::BufferPair kOutput = {kOutputMoverA, kOutputMoverB};

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kInputMoverA, kOutputMoverA);
    compute.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler != kHandlerRead && handler != kHandlerCopy) {
            return;
        }
        const uint32_t tiles = command.a2;
        const uint32_t tiles_per_block = command.a3;
        const km::Pattern pattern = km::Pattern::of(command.a4);
        if (!pattern.valid() || tiles_per_block == 0) {
            return;
        }
        const bool copy = handler == kHandlerCopy;
        const uint32_t blocks = (
            (tiles + tiles_per_block - 1) / tiles_per_block * (copy ? 1 : command.a5)
        );
        for (uint32_t block = 0; block < blocks; block++) {
            const uint32_t input = km::block_half(kInput, pattern, block);
            if (!compute.wait_front(input, tiles_per_block)) {
                return;
            }
            if (copy) {
                // A ragged last block is copied whole; the movers write only its pages.
                const uint32_t output = km::block_half(kOutput, pattern, block);
                const bool room = compute.reserve_back(output, tiles_per_block);
                copy_init(input);
                for (uint32_t tile = 0; tile < tiles_per_block; tile++) {
                    tile_regs_acquire();
                    copy_tile(input, tile, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    if (room) {
                        pack_tile<true>(0, output, tile);
                    }
                    tile_regs_release();
                }
                if (room) {
                    cb_push_back(output, tiles_per_block);
                }
            }
            cb_pop_front(input, tiles_per_block);
        }
    });
}
