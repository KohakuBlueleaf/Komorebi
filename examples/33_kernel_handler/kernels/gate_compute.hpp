// The gate's compute (../../22_gate/kernels/compute.cpp) as the hand kernel of the
// class's handler `gate`: out = silu(a) * b, a block of tiles at a time.
#pragma once

#include "api/compute/eltwise_binary.h"

namespace gate_k {

    constexpr auto kDestToSourceA = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

    // GATE on the three compute CPUs: each block from the half of the pair its mover
    // filled, silu(a) * b in the vector unit, packed into the result pair's half.
    inline void gate_compute(
        km::Compute& compute,
        const km_command_t& command,
        uint32_t tiles,
        km::Pattern pattern
    ) {
        static_cast<void>(command);
        for (uint32_t block = 0; block < tiles / kConstBlock; block++) {
            const uint32_t a = km::block_half(kPairAIn, pattern, block);
            const uint32_t b = km::block_half(kPairBIn, pattern, block);
            const uint32_t out = km::block_half(kPairResult, pattern, block);
            if (
                !compute.wait_front(a, kConstBlock) ||
                !compute.wait_front(b, kConstBlock)
            ) {
                return;
            }
            // one pass over the block for each step: each step's init sets the unit up
            tile_regs_acquire();
            copy_init(a);
            for (uint32_t tile = 0; tile < kConstBlock; tile++) {
                copy_tile(a, tile, tile);  // a into the vector unit's registers
            }
            silu_tile_init();
            for (uint32_t tile = 0; tile < kConstBlock; tile++) {
                silu_tile(tile);  // silu(a)
            }
            mul_reuse_dest_init<kDestToSourceA>(b);
            for (uint32_t tile = 0; tile < kConstBlock; tile++) {
                mul_reuse_dest_tiles<kDestToSourceA>(b, tile, tile);  // * b
            }
            tile_regs_commit();
            tile_regs_wait();
            const bool room = compute.reserve_back(out, kConstBlock);
            for (uint32_t tile = 0; room && tile < kConstBlock; tile++) {
                pack_tile(tile, out);
            }
            tile_regs_release();
            if (!room) {
                return;
            }
            cb_push_back(out, kConstBlock);
            cb_pop_front(a, kConstBlock);
            cb_pop_front(b, kConstBlock);
        }
    }

}  // namespace gate_k
