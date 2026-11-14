// Linear writer, NoC1: the column broadcast of this island's W; Y (all of K) or the
// partial sums into P, every island told, then this core's share of Y summed.
#include <stdint.h>

#include "../dense.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "linear.h"

namespace {

    // Waits until port `port` of this core has counted `wanted` arrivals, then clears
    // it. False: gave up (recorded).
    bool counted(
        const km::Writer& writer,
        const km_command_t& command,
        uint32_t port,
        uint32_t wanted
    ) {
        volatile tt_l1_ptr uint32_t* words = km::port_words(writer, port);
        words[KM_PORT_WANT] = wanted;
        const uint32_t code = writer.poll_at(
            km::kWherePort,
            [&] {
                invalidate_l1_cache();
                return static_cast<int32_t>(words[KM_PORT_COUNT] - wanted) >= 0;
            },
            writer.args().wait_limit
        );
        if (code != km::kGaveNone) {
            writer.gave_up(
                km::kStWriter,
                code,
                km::kWherePort,
                km::slot(command) | port << 8,
                command.sequence
            );
            return false;
        }
        words[KM_PORT_COUNT] = 0;
        words[KM_PORT_WANT] = 0;
        return true;
    }

    // One token to port `port` of rank `rank`, wherever it is (another island's
    // through the relay).
    bool tell(
        const km::Writer& writer,
        const km_command_t& command,
        uint32_t rank,
        uint32_t port
    ) {
        writer.produce(kCbToken);
        cb_reserve_back(kCbToken, 1);
        cb_push_back(kCbToken, 1);
        const km::Arrival arrival = {kCbArrive, 1, 1};
        return (
            km::to_island(km::island_of(rank))
                .put(writer, command, arrival, kCbToken, 1, rank, port, 0)
        );
    }

    // Every island's partial sums are in its DRAM once this returns on every core:
    // up the island's tree, its leader to the other leaders, down the tree.
    bool everywhere(const km::Writer& writer, const km_command_t& command) {
        const uint32_t code = writer.poll_at(
            km::kWhereData,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            writer.args().wait_limit
        );
        if (code != km::kGaveNone) {
            writer.gave_up(
                km::kStWriter, code, km::kWhereData, km::slot(command), command.sequence
            );
            return false;
        }
        const km::Group group = km::group_of(command, writer.me());
        const km::Group local = group.local();
        const uint32_t rank = local.rank;
        const uint32_t children = (
            (2 * rank + 1 < local.size ? 1 : 0) + (2 * rank + 2 < local.size ? 1 : 0)
        );
        if (children != 0 && !counted(writer, command, kPortUp, children)) {
            return false;
        }
        if (rank != 0) {
            if (!tell(writer, command, km::core_of(local, (rank - 1) / 2), kPortUp)) {
                return false;
            }
            if (!counted(writer, command, kPortDown, 1)) {
                return false;
            }
        } else {
            const km::Group leaders = group.leaders();
            for (uint32_t part = 0; part < leaders.size; ++part) {
                if (
                    part != leaders.rank &&
                    !tell(writer, command, group.leader(part), kPortCross)
                ) {
                    return false;
                }
            }
            if (
                leaders.size > 1 &&
                !counted(writer, command, kPortCross, leaders.size - 1)
            ) {
                return false;
            }
        }
        for (uint32_t child = 2 * rank + 1; child < 2 * rank + 3; ++child) {
            if (
                child < local.size &&
                !tell(writer, command, km::core_of(local, child), kPortDown)
            ) {
                return false;
            }
        }
        return tell(writer, command, km::core_of(local, rank), kPortGo);
    }

}  // namespace

void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto y_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    constexpr auto p_args = (
        TensorAccessorArgs<y_args.next_compile_time_args_offset()>()
    );
    km::Writer writer;
    const auto core = [&](uint32_t word) {
        return writer.arg(kArgCore + word);
    };
    const auto plan = [&](uint32_t word) {
        return core(kCorePlan + word);
    };
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return true;
        }
        const uint32_t w = command.a1, y = command.a2, p = command.a3;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t m_tiles = plan(kPlMt), here = km::island();
        invalidate_l1_cache();  // the tensor table
        const Cover mine = cover(w, here, k_whole, n_whole);
        const bool whole = parts(w, k_whole, n_whole).whole;
        const Holding y_here = holding(y, here, m_tiles * n_whole);
        const auto y_matrix = TensorAccessor(y_args, y_here.address);
        const auto p_matrix = (
            TensorAccessor(p_args, holding(p, here, m_tiles * n_whole).address)
        );
        if (mine.k_tiles != 0 && core(kCoreRank) != 0) {
            km::Broadcast broadcast(writer, command, kArgBroadcast);
            if (!broadcast.ok()) {
                return false;
            }
            const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
            const uint32_t k_block_tiles = plan(kPlKb);
            const uint32_t k_blocks = blocks_of(mine.k_tiles, k_block_tiles);
            const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
            const uint32_t subblock_height = plan(kPlSh);
            const uint32_t subblock_width = plan(kPlSw);
            const uint32_t subblock_tiles = subblock_height * subblock_width;
            const uint32_t first_row = core(kCoreM0), first_column = core(kCoreN0);
            const uint32_t row_end = (
                first_row + plan(kPlPcM) < m_tiles ? first_row + plan(kPlPcM) : m_tiles
            );
            const uint32_t column_end =
                (  // of this island's columns of W and Y
                    first_column + plan(kPlPcN) < mine.n_tiles
                        ? first_column + plan(kPlPcN)
                        : mine.n_tiles
                );
            const Holding w_here = holding(w, here, k_whole * n_whole);
            const auto w_matrix = TensorAccessor(w_args, w_here.address);
            const uint32_t out = whole ? kCbYOut : kCbPartOut;
            const uint32_t out_bytes = whole ? kBf16Bytes : kPartBytes;

            // Output block (block_row, block_column), subblocks in compute's order: Y
            // where its pages are (here, else put across), or the partial sums into P.
            const auto put = [&](uint32_t block_row, uint32_t block_column) {
                for (
                    uint32_t subblock_row = 0; subblock_row < block_height;
                    subblock_row += subblock_height
                ) {
                    for (
                        uint32_t subblock_column = 0; subblock_column < block_width;
                        subblock_column += subblock_width
                    ) {
                        if (!writer.wait_front(out, subblock_tiles)) {
                            return false;
                        }
                        uint32_t source = get_read_ptr(out);
                        const uint32_t column = block_column + subblock_column;
                        const uint32_t left = (
                            column < column_end ? column_end - column : 0
                        );
                        for (
                            uint32_t tile_row = 0; tile_row < subblock_height;
                            ++tile_row
                        ) {
                            // the row's tiles: pages page .. of Y (or P), Y's here
                            // at consecutive indices when its part holds them all
                            const uint32_t row = block_row + subblock_row + tile_row;
                            const uint32_t columns = (
                                row >= row_end
                                    ? 0
                                    : left < subblock_width
                                        ? left
                                        : subblock_width
                            );
                            const uint32_t page = row * n_whole + mine.n0 + column;
                            const bool home = (
                                whole && columns != 0 && y_here.holds(page, columns)
                            );
                            const uint32_t index = home ? y_here.index(page) : 0;
                            for (
                                uint32_t tile_column = 0; tile_column < subblock_width;
                                ++tile_column
                            ) {
                                if (tile_column < columns) {
                                    if (!whole) {
                                        noc_async_write_page(
                                            page + tile_column, p_matrix, source
                                        );
                                    } else if (home) {
                                        noc_async_write_page(
                                            index + tile_column, y_matrix, source
                                        );
                                    } else if (
                                        km::island_of(y, page + tile_column) == here
                                    ) {
                                        noc_async_write_page(
                                            y_here.index(page + tile_column),
                                            y_matrix,
                                            source
                                        );
                                    } else if (!km::write_pages(
                                        writer,
                                        command,
                                        y_args,
                                        y,
                                        page + tile_column,
                                        1,
                                        source,
                                        kPortY
                                    )) {
                                        return false;
                                    }
                                }
                                source += out_bytes;
                            }
                        }
                        if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                            return false;
                        }
                        cb_pop_front(out, subblock_tiles);
                    }
                }
                return true;
            };

            bool pending = false;
            uint32_t pending_row = 0, pending_column = 0, step = 0;
            for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
                const uint32_t block_row = (
                    first_row + block / blocks_wide * block_height
                );
                const uint32_t block_column = (
                    first_column + block % blocks_wide * block_width
                );
                for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                    if (!broadcast.sender()) {
                        if (!broadcast.arrive(step)) {
                            return false;
                        }
                    } else {
                        const uint32_t base = broadcast.reserve(step);
                        if (base == 0) {
                            return false;
                        }
                        const uint32_t k_start = k_block * k_block_tiles;
                        const uint32_t k_length = (
                            mine.k_tiles - k_start < k_block_tiles
                                ? mine.k_tiles - k_start
                                : k_block_tiles
                        );
                        const uint32_t left =
                            (  // of this island's columns
                                block_column < mine.n_tiles
                                    ? mine.n_tiles - block_column
                                    : 0
                            );
                        const uint32_t columns = (
                            left < block_width ? left : block_width
                        );
                        for (uint32_t k_tile = 0; k_tile < k_length; ++k_tile) {
                            // W's rows this island holds, read here: a row's tiles
                            // at consecutive indices of its part
                            const uint32_t index = (
                                w_here.index(
                                    (mine.k0 + k_start + k_tile) * n_whole + mine.n0 +
                                    block_column
                                )
                            );
                            for (uint32_t column = 0; column < columns; ++column) {
                                noc_async_read_page(
                                    index + column,
                                    w_matrix,
                                    base + (k_tile * block_width + column) * kBf16Bytes
                                );
                            }
                        }
                        if (!flushed(writer, km::kStWriter, km::kFlushReads)) {
                            return false;
                        }
                        if (!broadcast.cast(step)) {
                            return false;
                        }
                    }
                    if (k_block == 0 && pending) {
                        if (!put(pending_row, pending_column)) {
                            return false;
                        }
                        pending = false;
                    }
                }
                pending = true;
                pending_row = block_row;
                pending_column = block_column;
            }
            if (pending && !put(pending_row, pending_column)) {
                return false;
            }
            if (!broadcast.finish()) {
                return false;
            }
        }
        if (whole) {
            return true;
        }
        if (!everywhere(writer, command)) {
            return false;
        }
        // Y's share of this core: the sums compute made of every island's part.
        const km::Group local = km::group_of(command, writer.me()).local();
        const km::Partition share = km::partition(y_here.pages, local.rank, local.size);
        for (uint32_t tile = 0; tile < share.len; ++tile) {
            if (!writer.wait_front(kCbYOut, 1, km::kBehindPort)) {
                return false;
            }
            noc_async_write_page(share.start + tile, y_matrix, get_read_ptr(kCbYOut));
            if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                return false;
            }
            cb_pop_front(kCbYOut, 1);
        }
        return true;
    });
}
