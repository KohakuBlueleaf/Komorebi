// The example's compute kernel: out = silu(a) * b, a block of tiles at a time, in the
// vector unit.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "gate.h"
#include "komorebi/device/km_movement.hpp"
#include "komorebi/device/km_worker.hpp"

constexpr km::BufferPair kA = {kAMoverA, kAMoverB};
constexpr km::BufferPair kB = {kBMoverA, kBMoverB};
constexpr km::BufferPair kOut = {kOutMoverA, kOutMoverB};
constexpr auto kDestToSourceA = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

void kernel_main() {
    km::Compute compute;
    compute_kernel_hw_startup(kAMoverA, kOutMoverA);
    compute.run([&](const km_command_t& command) {
        const km::Pattern pattern = km::Pattern::of(command.a4);
        for (uint32_t block = 0; block < command.a3 / kBlock; block++) {
            // each block from the half of the buffer pair its mover filled
            const uint32_t a = km::block_half(kA, pattern, block);
            const uint32_t b = km::block_half(kB, pattern, block);
            const uint32_t out = km::block_half(kOut, pattern, block);
            if (!compute.wait_front(a, kBlock) || !compute.wait_front(b, kBlock)) {
                return;
            }
            // one pass over the block for each step: each step's init sets the unit up
            tile_regs_acquire();
            copy_init(a);
            for (uint32_t tile = 0; tile < kBlock; tile++) {
                copy_tile(a, tile, tile);  // a into the vector unit's registers
            }
            silu_tile_init();
            for (uint32_t tile = 0; tile < kBlock; tile++) {
                silu_tile(tile);  // silu(a)
            }
            mul_reuse_dest_init<kDestToSourceA>(b);
            for (uint32_t tile = 0; tile < kBlock; tile++) {
                mul_reuse_dest_tiles<kDestToSourceA>(b, tile, tile);  // * b
            }
            tile_regs_commit();
            tile_regs_wait();
            const bool room = compute.reserve_back(out, kBlock);
            for (uint32_t tile = 0; room && tile < kBlock; tile++) {
                pack_tile(tile, out);
            }
            tile_regs_release();
            if (!room) {
                return;
            }
            cb_push_back(out, kBlock);
            cb_pop_front(a, kBlock);
            cb_pop_front(b, kBlock);
        }
    });
}
