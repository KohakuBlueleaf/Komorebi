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
    constexpr auto p_args = (
        TensorAccessorArgs<x_args.next_compile_time_args_offset()>()
    );
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
        const uint32_t x = command.a0, w = command.a1, y = command.a2, p = command.a3;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t here = km::island(), islands = km::islands();
        const uint32_t m_tiles = plan(kPlMt);
        invalidate_l1_cache();  // the tensor table
        const Cover mine = cover(w, here, k_whole, n_whole);
        const Parts held_by = parts(w, k_whole, n_whole);
        if (mine.k_tiles != 0 && core(kCoreRank) != 0) {
            km::Broadcast broadcast(reader, command, kArgBroadcast);
            if (!broadcast.ok()) {
                return;
            }
            const uint32_t block_height = plan(kPlBm);
            const uint32_t k_block_tiles = plan(kPlKb);
            const uint32_t k_blocks = blocks_of(mine.k_tiles, k_block_tiles);
            const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
            const uint32_t first_row = core(kCoreM0);
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
                            x_args,
                            x,
                            page,
                            k_length,
                            base + tile_row * k_block_tiles * kBf16Bytes
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
        // seen every island's partial sums written (P: all of Y's pages on each).
        const km::Group local = km::group_of(command, reader.me()).local();
        const uint32_t pages = m_tiles * n_whole;
        const Holding y_here = holding(y, here, pages);
        const km::Partition share = km::partition(y_here.pages, local.rank, local.size);
        if (!km::port_wait(reader, command, kPortGo, 1)) {
            return;
        }
        const auto p_matrix = TensorAccessor(p_args, holding(p, here, pages).address);
        const uint32_t bytes = p_matrix.get_aligned_page_size();
        const uint32_t sums = __builtin_popcount(held_by.holders);
        for (uint32_t done = 0; done < share.len; done += kChunk) {
            const uint32_t count = (
                share.len - done < kChunk ? share.len - done : kChunk
            );
            const uint32_t first = share.start + done;  // of Y's part here
            if (!reader.reserve(kCbSums, kChunk * sums)) {
                return;
            }
            // each holder's partial sums of the chunk, in island order: here read,
            // elsewhere asked a run of consecutive pages at a time
            uint32_t to = get_write_ptr(kCbSums), asked = 0, holder = here;
            for (uint32_t island = 0; island < islands; ++island) {
                if (!((held_by.holders >> island) & 1)) {
                    continue;
                }
                const uint32_t address = holding(p, island, pages).address;
                for (uint32_t tile = 0; tile < count;) {
                    const uint32_t page = y_here.page(first + tile);
                    uint32_t run = 1;
                    while (
                        tile + run < count &&
                        y_here.page(first + tile + run) == page + run
                    ) {
                        ++run;
                    }
                    if (island == here) {
                        for (uint32_t k = 0; k < run; ++k) {
                            noc_async_read_page(
                                page + k, p_matrix, to + (tile + k) * bytes
                            );
                        }
                    } else if (!km::ask_pages(
                        reader,
                        command,
                        island,
                        address,
                        bytes,
                        page,
                        run,
                        to + tile * bytes
                    )) {
                        return;
                    } else {
                        asked += run;
                        holder = island;
                    }
                    tile += run;
                }
                to += kChunk * bytes;
            }
            if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                return;
            }
            if (
                asked != 0 &&
                !km::ask_wait(
                    reader,
                    km::slot(command),
                    command.sequence,
                    km::pages_port(reader),
                    asked,
                    holder
                )
            ) {
                return;
            }
            cb_push_back(kCbSums, kChunk * sums);
        }
    });
}
