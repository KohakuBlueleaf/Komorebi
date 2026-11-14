// Linear reader, NoC0: X's row broadcast over this island's K a window of columns; in
// its own window, the subblocks other islands put into S, to compute.
#include <stdint.h>

#include "../dense.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "linear.h"

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto s_args = (
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
        const uint32_t x = command.a0, w = command.a1, y = command.a2, s = command.a3;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t here = km::island(), islands = km::islands();
        const uint32_t m_tiles = plan(kPlMt);
        invalidate_l1_cache();  // the tensor table
        const Span mine = k_part(w, here, k_whole, n_whole);
        const bool exchange = exchanging(w, y, m_tiles, k_whole, n_whole);
        const bool included = core(kCoreRank) != 0;
        const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
        const uint32_t k_block_tiles = plan(kPlKb);
        const uint32_t k_blocks = blocks_of(mine.count, k_block_tiles);
        const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
        const uint32_t first_row = core(kCoreM0), first_column = core(kCoreN0);
        const uint32_t row_end = (
            first_row + plan(kPlPcM) < m_tiles ? first_row + plan(kPlPcM) : m_tiles
        );
        const uint32_t subblock_tiles = plan(kPlSh) * plan(kPlSw);
        const uint32_t block_tiles = block_height * block_width;
        const Sent sent = {core(kCoreSetPages), core(kCoreSent)};
        // the other islands that put here, a K block of a block each: k blocks 0 ..
        uint32_t senders = 0;
        for (uint32_t turn = 1; turn < islands; ++turn) {
            senders += k_part(w, island_at(here, turn), k_whole, n_whole).count != 0;
        }
        const uint32_t kept = columns(y, here, m_tiles, n_whole).count;
        if (
            included && mine.count != 0 && kept != 0 && senders > 1 &&
            senders >= k_blocks
        ) {
            reader.gave_up(  // more islands put here than K blocks to take them in
                km::kStReader,
                km::kGaveBadItem,
                km::kWhereData,
                km::slot(command),
                command.sequence
            );
            return;
        }
        const Holding x_here = holding(x, here, m_tiles * k_whole);
        const auto x_matrix = TensorAccessor(x_args, x_here.address);
        const auto s_matrix = TensorAccessor(s_args, holding(s, here, 1).address);
        // the writer's go, once a command: every island's puts are in S
        bool waited = !exchange;
        if (included && mine.count != 0) {
            km::Broadcast broadcast(reader, command, kArgBroadcast);
            if (!broadcast.ok()) {
                return;
            }
            uint32_t step = 0;
            for (uint32_t turn = 1; turn <= islands; ++turn) {
                const uint32_t to = island_at(here, turn);
                const uint32_t width = columns(y, to, m_tiles, n_whole).count;
                if (width == 0) {
                    continue;
                }
                const bool own = to == here;
                if (own && !waited) {
                    if (!km::port_wait(reader, command, kPortGo, 1)) {
                        return;
                    }
                    waited = true;
                }
                const uint32_t column_end = (
                    first_column + plan(kPlPcN) < width
                        ? first_column + plan(kPlPcN)
                        : width
                );
                for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
                    const uint32_t block_row = (
                        first_row + block / blocks_wide * block_height
                    );
                    const uint32_t block_column = (
                        first_column + block % blocks_wide * block_width
                    );
                    const bool inside = (
                        block_row < row_end && block_column < column_end
                    );
                    for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                        if (!broadcast.sender()) {
                            if (!broadcast.arrive(step)) {
                                return;
                            }
                        } else {
                            const uint32_t base = broadcast.reserve(step);
                            if (base == 0) {
                                return;
                            }
                            const uint32_t k_start = k_block * k_block_tiles;
                            const uint32_t k_length = (
                                mine.count - k_start < k_block_tiles
                                    ? mine.count - k_start
                                    : k_block_tiles
                            );
                            const uint32_t rows = (
                                block_row >= m_tiles
                                    ? 0
                                    : m_tiles - block_row < block_height
                                        ? m_tiles - block_row
                                        : block_height
                            );
                            for (uint32_t tile_row = 0; tile_row < rows; ++tile_row) {
                                // a row of X: this island's K, its part's pages
                                const uint32_t page = (
                                    (block_row + tile_row) * k_whole + mine.first +
                                    k_start
                                );
                                if (!x_here.holds(page, k_length)) {
                                    reader.gave_up(  // X is not split as W is
                                        km::kStReader,
                                        km::kGaveBadItem,
                                        km::kWhereData,
                                        km::slot(command),
                                        command.sequence
                                    );
                                    return;
                                }
                                const uint32_t index = x_here.index(page);
                                const uint32_t at = (
                                    base + tile_row * k_block_tiles * kBf16Bytes
                                );
                                for (uint32_t k = 0; k < k_length; ++k) {
                                    noc_async_read_page(
                                        index + k, x_matrix, at + k * kBf16Bytes
                                    );
                                }
                            }
                            if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                                return;
                            }
                            if (!broadcast.cast(step)) {
                                return;
                            }
                        }
                        if (!own || k_block >= senders) {
                            continue;
                        }
                        // sender k_block's tiles of this block, a subblock at a time
                        // (a block wholly outside the columns: none were put)
                        const uint32_t first_page = (
                            sent.page(k_block, block * block_tiles)
                        );
                        for (
                            uint32_t tile = 0; tile < block_tiles;
                            tile += subblock_tiles
                        ) {
                            if (!reader.reserve(kCbSent, subblock_tiles)) {
                                return;
                            }
                            const uint32_t at = get_write_ptr(kCbSent);
                            for (uint32_t k = 0; inside && k < subblock_tiles; ++k) {
                                noc_async_read_page(
                                    first_page + tile + k, s_matrix, at + k * kBf16Bytes
                                );
                            }
                            if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                                return;
                            }
                            cb_push_back(kCbSent, subblock_tiles);
                        }
                    }
                }
            }
            if (!broadcast.finish()) {
                return;
            }
        }
        if (!waited) {
            km::port_wait(reader, command, kPortGo, 1);
        }
    });
}
