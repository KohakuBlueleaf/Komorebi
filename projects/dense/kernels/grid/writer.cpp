// Grid writer: the column broadcast on NoC1 (W blocks, kb x bn tiles, multicast up the
// column) and Y: a block is written once the next block's first W block is delivered.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "grid.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    using W = km::Tensor<0>;
    using Y = km::Tensor<W::next_compile_time_args_offset()>;
    km::Writer writer;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbGridProf));
    const auto core = [&](uint32_t word) {
        return writer.arg(kArgCore + word);
    };
    const auto plan = [&](uint32_t word) {
        return core(kCorePlan + word);
    };
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerGridProf) {
            const uint32_t profile_address = command.a0;
            using OwnBank = InterleavedAddrGen<true>;  // km: own island
            const OwnBank pages = {
                .bank_base_address = profile_address, .page_size = kProfBytes
            };
            noc_async_write(
                get_write_ptr(kCbGridProf), get_noc_addr(writer.me(), pages), kProfBytes
            );
            return flushed(writer, km::kStWriter, km::kFlushSent);
        }
        if (km::handler(command) != kHandlerGrid) {
            return true;
        }
        const uint32_t started = km::wall();
        const uint32_t w_address = command.a1, y_address = command.a2;
        const uint32_t mode = command.a3;
        const bool broadcasts = mode != kGridProduct;
        const bool store = mode == kGridFull;
        km::Broadcast broadcast(writer, command, kArgBroadcast);
        if (!broadcast.ok()) {
            return false;
        }
        if (core(kCoreRank) == 0) {
            return true;
        }
        const uint32_t m_tiles = plan(kPlMt);
        const uint32_t k_tiles = plan(kPlKt);
        const uint32_t n_tiles = plan(kPlNt);
        const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
        const uint32_t k_block_tiles = plan(kPlKb);
        const uint32_t k_blocks = plan(kPlNk), blocks_high = plan(kPlNbh);
        const uint32_t blocks_wide = plan(kPlNbw);
        const uint32_t subblock_height = plan(kPlSh), subblock_width = plan(kPlSw);
        const uint32_t subblock_tiles = subblock_height * subblock_width;
        const uint32_t first_row = core(kCoreM0), first_column = core(kCoreN0);
        const uint32_t row_end = (
            first_row + plan(kPlPcM) < m_tiles ? first_row + plan(kPlPcM) : m_tiles
        );
        const uint32_t column_end = (
            first_column + plan(kPlPcN) < n_tiles
                ? first_column + plan(kPlPcN)
                : n_tiles
        );
        const W w_matrix(w_address);
        const Y y_matrix(y_address);
        uint32_t read = 0, broadcast_cycles = 0, wait = 0, write = 0;

        // Output block (block_row, block_column) from kCbGridOut, subblocks of
        // subblock_height x subblock_width tiles in compute's order.
        const auto put = [&](uint32_t block_row, uint32_t block_column) {
            for (
                uint32_t subblock_row = 0; subblock_row < block_height;
                subblock_row += subblock_height
            ) {
                for (
                    uint32_t subblock_column = 0; subblock_column < block_width;
                    subblock_column += subblock_width
                ) {
                    const uint32_t asked = km::wall();
                    if (!writer.wait_front(kCbGridOut, subblock_tiles)) {
                        return false;
                    }
                    const uint32_t ready = km::wall();
                    uint32_t source = get_read_ptr(kCbGridOut);
                    for (
                        uint32_t tile_row = 0; tile_row < subblock_height; ++tile_row
                    ) {
                        for (
                            uint32_t tile_column = 0; tile_column < subblock_width;
                            ++tile_column
                        ) {
                            const uint32_t row = block_row + subblock_row + tile_row;
                            const uint32_t column = (
                                block_column + subblock_column + tile_column
                            );
                            if (
                                store && row < row_end && column < column_end &&
                                !km::write_pages(
                                    writer,
                                    command,
                                    y_matrix,
                                    row * n_tiles + column,
                                    1,
                                    source
                                )
                            ) {
                                return false;
                            }
                            source += kTileBytes;
                        }
                    }
                    if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    cb_pop_front(kCbGridOut, subblock_tiles);
                    wait += ready - asked;
                    write += km::wall() - ready;
                }
            }
            return true;
        };

        bool pending = false;
        uint32_t pending_row = 0, pending_column = 0, step = 0;
        for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
            const uint32_t block_row = first_row + block / blocks_wide * block_height;
            const uint32_t block_column = (
                first_column + block % blocks_wide * block_width
            );
            for (
                uint32_t k_block = 0; broadcasts && k_block < k_blocks;
                ++k_block, ++step
            ) {
                const uint32_t step_started = km::wall();
                if (!broadcast.sender()) {
                    if (!broadcast.arrive(step)) {
                        return false;
                    }
                    broadcast_cycles += km::wall() - step_started;
                } else {
                    const uint32_t base = broadcast.reserve(step);
                    if (base == 0) {
                        return false;
                    }
                    const uint32_t reserved = km::wall();
                    const uint32_t k_start = k_block * k_block_tiles;
                    const uint32_t k_length = (
                        k_tiles - k_start < k_block_tiles
                            ? k_tiles - k_start
                            : k_block_tiles
                    );
                    const uint32_t columns = (
                        block_column >= n_tiles
                            ? 0
                            : n_tiles - block_column < block_width
                                ? n_tiles - block_column
                                : block_width
                    );
                    for (uint32_t k_tile = 0; k_tile < k_length; ++k_tile) {
                        const uint32_t page = (
                            (k_start + k_tile) * n_tiles + block_column
                        );
                        const uint32_t to = base + k_tile * block_width * kTileBytes;
                        if (!km::read_pages(
                            writer, command, w_matrix, page, columns, to
                        )) {
                            return false;
                        }
                    }
                    if (!flushed(writer, km::kStWriter, km::kFlushReads)) {
                        return false;
                    }
                    const uint32_t read_done = km::wall();
                    if (!broadcast.cast(step)) {
                        return false;
                    }
                    read += read_done - reserved;
                    broadcast_cycles += (
                        reserved - step_started + km::wall() - read_done
                    );
                }
                if (k_block == 0 && pending) {
                    if (!put(pending_row, pending_column)) {
                        return false;
                    }
                    pending = false;
                }
            }
            if (!broadcasts && pending && !put(pending_row, pending_column)) {
                return false;
            }
            pending = mode != kGridBroadcasts;
            pending_row = block_row;
            pending_column = block_column;
        }
        if (pending && !put(pending_row, pending_column)) {
            return false;
        }
        if (broadcasts && !broadcast.finish()) {
            return false;
        }
        km::add_sat(profile + kGpWrRead, read);
        km::add_sat(profile + kGpWrBroadcast, broadcast_cycles);
        km::add_sat(profile + kGpWrWait, wait);
        km::add_sat(profile + kGpWrWrite, write);
        km::add_sat(profile + kGpWrBody, km::wall() - started);
        return true;
    });
}
