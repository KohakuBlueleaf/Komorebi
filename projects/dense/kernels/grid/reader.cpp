// Grid reader: the row broadcast, on NoC0. The row's sender reads each A block (bm x kb
// tiles, row stride kb) from DRAM and multicasts it along the row; the others take it.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "grid.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    using A = km::Tensor<0>;
    km::Reader reader;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbGridProf));
    for (uint32_t word = 0; word < kGpWords; ++word) {
        profile[word] = 0;
    }
    const auto core = [&](uint32_t word) {
        return reader.arg(kArgCore + word);
    };
    const auto plan = [&](uint32_t word) {
        return core(kCorePlan + word);
    };
    reader.run([&](const km_command_t& command) {
        const uint32_t mode = command.a3;
        if (km::handler(command) != kHandlerGrid || mode == kGridProduct) {
            return;
        }
        const uint32_t started = km::wall();
        km::Broadcast broadcast(reader, command, kArgBroadcast);
        if (!broadcast.ok() || !broadcast.included()) {
            return;
        }
        const uint32_t a_address = command.a0;
        const uint32_t m_tiles = plan(kPlMt), k_tiles = plan(kPlKt);
        const uint32_t block_height = plan(kPlBm);
        const uint32_t k_block_tiles = plan(kPlKb), k_blocks = plan(kPlNk);
        const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
        const uint32_t first_row = core(kCoreM0);
        const A a_matrix(a_address);
        uint32_t read = 0, broadcast_cycles = 0, step = 0;
        for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
            const uint32_t block_row = first_row + block / blocks_wide * block_height;
            for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                const uint32_t step_started = km::wall();
                if (!broadcast.sender()) {
                    if (!broadcast.arrive(step)) {
                        return;
                    }
                    broadcast_cycles += km::wall() - step_started;
                    continue;
                }
                const uint32_t base = broadcast.reserve(step);
                if (base == 0) {
                    return;
                }
                const uint32_t reserved = km::wall();
                const uint32_t k_start = k_block * k_block_tiles;
                const uint32_t k_length = (
                    k_tiles - k_start < k_block_tiles
                        ? k_tiles - k_start
                        : k_block_tiles
                );
                for (
                    uint32_t tile_row = 0;
                    tile_row < block_height && block_row + tile_row < m_tiles;
                    ++tile_row
                ) {
                    const uint32_t page = (block_row + tile_row) * k_tiles + k_start;
                    const uint32_t to = base + tile_row * k_block_tiles * kTileBytes;
                    if (!km::read_pages(
                        reader, command, a_matrix, page, k_length, to
                    )) {
                        return;
                    }
                }
                if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                    return;
                }
                const uint32_t read_done = km::wall();
                if (!broadcast.cast(step)) {
                    return;
                }
                read += read_done - reserved;
                broadcast_cycles += reserved - step_started + km::wall() - read_done;
            }
        }
        if (!broadcast.finish()) {
            return;
        }
        km::add_sat(profile + kGpRdRead, read);
        km::add_sat(profile + kGpRdBroadcast, broadcast_cycles);
        km::add_sat(profile + kGpRdBody, km::wall() - started);
    });
}
