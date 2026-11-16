// The probe's compute kernel: reads take block k from the half its pattern names (a
// checked run sums every word), writes hand block k's pages to that half.
#include <stdint.h>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "komorebi/device/movement/km_movement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "probe.h"

namespace {

    // Adds the words of the `pages` pages at the front of `buffer` to the sums; the
    // weighted sum counts tile `position` (from 0) position + 1 times.
    void add_words(
        uint32_t buffer,
        uint32_t pages,
        uint32_t position,
        uint32_t& sum,
        uint32_t& weighted
    ) {
#if defined(TRISC_UNPACK)
        for (uint32_t tile = 0; tile < pages; tile++) {
            const uint32_t address = ckernel::get_tile_l1_byte_address(buffer, tile);
            volatile tt_l1_ptr uint32_t* words = km::l1(address);
            uint32_t tile_sum = 0;
            for (uint32_t word = 0; word < kTileBytes / 4; word++) {
                tile_sum += words[word];
            }
            sum += tile_sum;
            weighted += tile_sum * (position + tile + 1);
        }
#else
        (void)buffer, (void)pages, (void)position, (void)sum, (void)weighted;
#endif
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_tile_address(kBufferProfile, 0));
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRun) {
            return;
        }
        const uint32_t tiles = command.a2;
        const uint32_t tiles_per_block = command.a3;
        const uint32_t pattern_word = command.a4;
        const uint32_t repetitions = command.a5;
        const uint32_t program = command.a6;
        const uint32_t salt = command.a8;
        if (program == kProgramCopy || tiles_per_block == 0) {
            return;
        }
        const km::Pattern pattern = km::Pattern::of(pattern_word);
        if (!pattern.valid()) {
            return;
        }
        const uint32_t blocks = tiles / tiles_per_block * repetitions;
        const uint32_t start = km::wall();
        uint32_t sum = 0, weighted = 0, taken = 0;
        for (uint32_t block = 0; block < blocks; block++) {
            const uint32_t buffer = km::block_half(
                {kBufferA, kBufferB}, pattern, block
            );
            if (program == kProgramRead) {
                if (!compute.wait_front(buffer, tiles_per_block)) {
                    return;
                }
                if (salt != 0) {
                    add_words(
                        buffer, tiles_per_block, block * tiles_per_block, sum, weighted
                    );
                }
                cb_pop_front(buffer, tiles_per_block);
            } else {
                if (!compute.reserve_back(buffer, tiles_per_block)) {
                    return;
                }
                cb_push_back(buffer, tiles_per_block);
            }
            taken += tiles_per_block;
        }
        UNPACK(({
            profile[kProfileComputeStart] = start;
            profile[kProfileComputeEnd] = km::wall();
            profile[kProfileSum] = sum;
            profile[kProfileWeighted] = weighted;
            profile[kProfileComputeTiles] = taken;
        }));
    });
}
