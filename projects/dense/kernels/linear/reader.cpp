// Linear reader, NoC0: X's row broadcast over this island's K (km::read_pages); for
// part of K, this core's share of Y's partial sums: its island's, the others' asked.
#include <stdint.h>

#include "../dense.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "linear.h"

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr uint32_t kXPlace = x_args.next_compile_time_args_offset();
    constexpr auto p_args = (
        TensorAccessorArgs< km::Placement<kXPlace>::next_compile_time_args_offset()>()
    );
    constexpr uint32_t kYPlace = p_args.next_compile_time_args_offset();
    constexpr uint32_t kWPlace = (
        km::Placement<kYPlace>::next_compile_time_args_offset()
    );
    const km::Placement<kXPlace> x_place{};
    km::Reader reader;
    const auto core = [&](uint32_t word) {
        return reader.arg(kArgCore + word);
    };
    const auto plan = [&](uint32_t word) {
        return core(kCorePlan + word);
    };
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return;
        }
        const uint32_t x_address = command.a0, p_address = command.a3;
        const uint32_t k_whole = command.a4;
        const uint32_t here = km::island(), islands = km::islands();
        const Cover mine = cover<kWPlace>(here, plan(kPlNt));
        const Parts held_by = parts<kWPlace>(k_whole, plan(kPlNt));
        if (mine.k_tiles != 0 && core(kCoreRank) != 0) {
            km::Broadcast broadcast(reader, command, kArgBroadcast);
            if (!broadcast.ok()) {
                return;
            }
            const uint32_t m_tiles = plan(kPlMt);
            const uint32_t block_height = plan(kPlBm);
            const uint32_t k_block_tiles = plan(kPlKb);
            const uint32_t k_blocks = blocks_of(mine.k_tiles, k_block_tiles);
            const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
            const uint32_t first_row = core(kCoreM0);
            const auto x_matrix = TensorAccessor(x_args, x_address);
            uint32_t step = 0;
            for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
                const uint32_t block_row = (
                    first_row + block / blocks_wide * block_height
                );
                for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                    if (!broadcast.sender()) {
                        if (!broadcast.arrive(step)) {
                            return;
                        }
                        continue;
                    }
                    const uint32_t base = broadcast.reserve(step);
                    if (base == 0) {
                        return;
                    }
                    const uint32_t k_start = k_block * k_block_tiles;
                    const uint32_t k_length = (
                        mine.k_tiles - k_start < k_block_tiles
                            ? mine.k_tiles - k_start
                            : k_block_tiles
                    );
                    for (
                        uint32_t tile_row = 0;
                        tile_row < block_height && block_row + tile_row < m_tiles;
                        ++tile_row
                    ) {
                        // a row of X where it lives: here, or asked of its island
                        const uint32_t page = (
                            (block_row + tile_row) * k_whole + mine.k0 + k_start
                        );
                        if (!km::read_pages(
                            reader,
                            command,
                            x_place,
                            x_matrix,
                            x_address,
                            page,
                            k_length,
                            base + tile_row * k_block_tiles * kBf16Bytes,
                            kPortX
                        )) {
                            return;
                        }
                    }
                    if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                        return;
                    }
                    if (!broadcast.cast(step)) {
                        return;
                    }
                }
            }
            if (!broadcast.finish()) {
                return;
            }
        }
        if (held_by.whole) {  // one island's product is Y itself
            return;
        }
        // This core's share of the Y pages this island holds, once the writer has
        // seen every island's partial sums written.
        const km::Group local = km::group_of(command, reader.me()).local();
        const Span y_here = held<kYPlace>(km::island());
        const km::Partition share = km::partition(y_here.pages, local.rank, local.size);
        if (!km::port_wait(reader, command, kPortGo, 1)) {
            return;
        }
        const auto p_matrix = TensorAccessor(p_args, p_address);
        const uint32_t bytes = p_matrix.get_aligned_page_size();
        const uint32_t sums = __builtin_popcount(held_by.holders);
        for (uint32_t done = 0; done < share.len; done += kChunk) {
            const uint32_t count = (
                share.len - done < kChunk ? share.len - done : kChunk
            );
            const uint32_t first = y_here.first + share.start + done;
            if (!reader.reserve(kCbSums, kChunk * sums)) {
                return;
            }
            // each holder's partial sums of the chunk, in island order: here read,
            // elsewhere asked
            uint32_t to = get_write_ptr(kCbSums), asked = 0;
            for (uint32_t island = 0; island < islands; ++island) {
                if (!((held_by.holders >> island) & 1)) {
                    continue;
                }
                if (island == here) {
                    for (uint32_t tile = 0; tile < count; ++tile) {
                        noc_async_read_page(first + tile, p_matrix, to + tile * bytes);
                    }
                } else if (!km::ask_pages(
                    reader,
                    command,
                    island,
                    p_address,
                    bytes,
                    first,
                    count,
                    to,
                    kPortP
                )) {
                    return;
                } else {
                    asked += count;
                }
                to += kChunk * bytes;
            }
            if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                return;
            }
            if (asked != 0 && !km::port_wait(reader, command, kPortP, asked)) {
                return;
            }
            cb_push_back(kCbSums, kChunk * sums);
        }
    });
}
