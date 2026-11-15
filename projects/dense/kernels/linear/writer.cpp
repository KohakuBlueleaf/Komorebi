// Linear writer (NoC1 by default): W's column broadcast over this island's K, its
// own window of Y written (or, LINEAR_ADD, its partial added with the other's).
#include <stddef.h>
#include <stdint.h>

#include "../dense.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "linear.h"

#if defined(LINEAR_SPLIT)  // the K-split plan (linear.h): its own body
#include "ksplit_writer.hpp"
#else

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

    // Waits until word COUNT of port `port` of this core holds `marker`. False: gave
    // up (recorded).
    bool marked(
        const km::Writer& writer,
        const km_command_t& command,
        uint32_t port,
        uint32_t marker
    ) {
        volatile tt_l1_ptr uint32_t* words = km::port_words(writer, port);
        const uint32_t code = writer.poll_at(
            km::kWherePort,
            [&] {
                invalidate_l1_cache();
                return words[KM_PORT_COUNT] == marker;
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

    // This CPU's writes have landed (acknowledged), so other cores may read them from
    // DRAM. False: gave up (recorded).
    bool landed(const km::Writer& writer, const km_command_t& command) {
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
        return true;
    }

}  // namespace

void kernel_main() {
    constexpr auto w_args = TensorAccessorArgs<0>();
    constexpr auto y_args = (
        TensorAccessorArgs<w_args.next_compile_time_args_offset()>()
    );
    constexpr auto t_args = (
        TensorAccessorArgs<y_args.next_compile_time_args_offset()>()
    );
    constexpr auto s_args = (
        TensorAccessorArgs<t_args.next_compile_time_args_offset()>()
    );
    km::Writer writer;
    const auto core = [&](uint32_t word) {
        return writer.arg(kArgCore + word);
    };
    const auto plan = [&](uint32_t word) {
        return core(kCorePlan + word);
    };
    uint32_t parity = 0;  // LINEAR_ADD: the command's set of S
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return true;
        }
        volatile tt_l1_ptr uint32_t* stamps = km::l1(get_write_ptr(kCbStamp));
        stamps[kStampStart] = km::wall();
        stamps[kStampPut] = 0;  // 0: no addition across this command
        stamps[kStampSynced] = 0;
        stamps[kStampSettled] = 0;
        stamps[kStampUp] = 0;
        stamps[kStampDown] = 0;
        LINEAR_ACCOUNT_DO(uint32_t account[kAcYWrite - kAcWFree + 1] = {});
        const uint32_t w = command.a1, y = command.a2, s = command.a3;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t m_tiles = plan(kPlMt), here = km::island();
        const uint32_t islands = km::islands();
        invalidate_l1_cache();  // the tensor table
        const Span mine = k_part(w, here, k_whole, n_whole);
        const uint32_t x = command.a0;
        // the rows this island makes: the plan's rows start at its first
        const Span made = made_rows(x, here, m_tiles, k_whole);
        const Holding y_here = holding(y, here, m_tiles * n_whole);
        const auto y_matrix = TensorAccessor(y_args, y_here.address);
        // the input-dim-sharded form's addition here: two islands, Y a copy
        const bool adding = LINEAR_ADDS && islands == 2 && copied(y);
        const auto s_matrix = TensorAccessor(s_args, holding(s, here, 1).address);
        const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
        const uint32_t block_tiles = block_height * block_width;
        if (mine.count != 0 && core(kCoreRank) != 0) {
            km::Broadcast broadcast(writer, command, kArgBroadcast);
#if defined(LINEAR_VALID_SENT)
            broadcast.valid_when_sent();
#endif
            if (!broadcast.ok()) {
                return false;
            }
            const uint32_t k_block_tiles = plan(kPlKb);
            const uint32_t k_blocks = blocks_of(mine.count, k_block_tiles);
            const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
            const uint32_t subblock_height = plan(kPlSh);
            const uint32_t subblock_width = plan(kPlSw);
            const uint32_t subblock_tiles = subblock_height * subblock_width;
            const uint32_t first_row = made.first + core(kCoreM0);
            const uint32_t first_column = core(kCoreN0);
            const uint32_t made_end = made.first + made.count;
            const uint32_t row_end = (
                first_row + plan(kPlPcM) < made_end
                    ? first_row + plan(kPlPcM)
                    : made_end
            );
            const Sent sent = {core(kCoreSetPages), core(kCoreSent)};
            const Holding w_here = holding(w, here, k_whole * n_whole);
            const auto w_matrix = TensorAccessor(w_args, w_here.address);
            // LINEAR_SPREAD: the column broadcast's ranks (sender 0), when a receiver's
            // rows of a block fit its slice buffer; the sender's slots, multicast area
            const auto cast_arg = [&](uint32_t offset) {
                return writer.arg(kArgBroadcast + offset / 4);
            };
            const uint32_t column_ranks = (
                cast_arg(offsetof(km::BroadcastArgs, receivers)) + 1
            );
            const uint32_t mc_start = cast_arg(offsetof(km::BroadcastArgs, mc_start));
            const uint32_t mc_end = cast_arg(offsetof(km::BroadcastArgs, mc_end));
            const LocalCBInterface& slice_cb = get_local_cb_interface(kCbWSlice);
            const LocalCBInterface& column_cb = get_local_cb_interface(kCbCol);
            const uint32_t block_bytes = broadcast.pages() * column_cb.fifo_page_size;
            const bool spread = (
                LINEAR_SPREADS && broadcast.included() && column_ranks > 1 &&
                blocks_of(k_block_tiles, column_ranks) * plan(kPlBn) * kBf16Bytes <=
                    slice_cb.fifo_size
            );
            const uint32_t slice_base = slice_cb.fifo_limit - slice_cb.fifo_size;
            const uint32_t port_base = writer.args().control + KM_CONTROL_PORT;
            uint32_t step = 0;
            bool finished = false;  // the column broadcast (LINEAR_ADD: before the sum)
            // this island's own columns of the rows it makes (none: nothing to make)
            const Span window = (
                ::window(x, w, y, here, here, m_tiles, k_whole, n_whole)
            );
            if (window.count != 0) {
                const uint32_t column_end = (
                    first_column + plan(kPlPcN) < window.count
                        ? first_column + plan(kPlPcN)
                        : window.count
                );
                // Output block `block` at (block_row, block_column) made: its
                // window's Y written by subblock.
                const auto put = [&](
                    uint32_t block, uint32_t block_row, uint32_t block_column
                ) {
                    for (
                        uint32_t subblock_row = 0; subblock_row < block_height;
                        subblock_row += subblock_height
                    ) {
                        for (
                            uint32_t subblock_column = 0; subblock_column < block_width;
                            subblock_column += subblock_width
                        ) {
                            LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                            if (!writer.wait_front(kCbYOut, subblock_tiles)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcYWait - 16], since));
                            uint32_t source = get_read_ptr(kCbYOut);
                            const uint32_t column = block_column + subblock_column;
                            const uint32_t left = (
                                column < column_end ? column_end - column : 0
                            );
                            for (
                                uint32_t tile_row = 0; tile_row < subblock_height;
                                ++tile_row
                            ) {
                                // the row's tiles: Y's pages here, consecutive
                                const uint32_t row = (
                                    block_row + subblock_row + tile_row
                                );
                                const uint32_t columns_here = (
                                    row >= row_end
                                        ? 0
                                        : left < subblock_width
                                            ? left
                                            : subblock_width
                                );
                                const uint32_t index = (
                                    columns_here == 0
                                        ? 0
                                        : y_here.index(
                                            row * n_whole + window.first + column
                                        )
                                );
                                for (
                                    uint32_t tile_column = 0;
                                    tile_column < subblock_width;
                                    ++tile_column
                                ) {
                                    if (tile_column < columns_here) {
                                        noc_async_write_page(
                                            index + tile_column, y_matrix, source
                                        );
                                    }
                                    source += kBf16Bytes;
                                }
                            }
                            if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcYWrite - 16], since));
                            cb_pop_front(kCbYOut, subblock_tiles);
                        }
                    }
                    return true;
                };

                // LINEAR_ADD: output block `block` made, its partial to this island's
                // set of S, a subblock at a time.
                const auto to_s = [&](uint32_t block) {
                    for (
                        uint32_t tile = 0; tile < block_tiles; tile += subblock_tiles
                    ) {
                        LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                        if (!writer.wait_front(kCbYOut, subblock_tiles)) {
                            return false;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[kAcYWait - 16], since));
                        const uint32_t first = (
                            sent.page(add_set(parity, here), block * block_tiles + tile)
                        );
                        const uint32_t source = get_read_ptr(kCbYOut);
                        for (uint32_t k = 0; k < subblock_tiles; ++k) {
                            noc_async_write_page(
                                first + k, s_matrix, source + k * kBf16Bytes
                            );
                        }
                        if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                            return false;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[kAcYWrite - 16], since));
                        cb_pop_front(kCbYOut, subblock_tiles);
                    }
                    return true;
                };
                const auto made = [&](uint32_t block, uint32_t row, uint32_t column) {
                    return adding ? to_s(block) : put(block, row, column);
                };

                bool pending = false;
                uint32_t pending_block = 0, pending_row = 0, pending_column = 0;
                for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
                    const uint32_t block_row = (
                        first_row + block / blocks_wide * block_height
                    );
                    const uint32_t block_column = (
                        first_column + block % blocks_wide * block_width
                    );
                    const uint32_t left =
                        (  // of the window's columns
                            block_column < window.count
                                ? window.count - block_column
                                : 0
                        );
                    const uint32_t columns_read = (
                        left < block_width ? left : block_width
                    );
                    // W's rows of K block `k_block` this rank reads (all on the
                    // sender, unless spread): (first row of the block, rows)
                    const auto k_rows = [&](uint32_t k_block) {
                        const uint32_t k_start = k_block * k_block_tiles;
                        return (
                            mine.count - k_start < k_block_tiles
                                ? mine.count - k_start
                                : k_block_tiles
                        );
                    };
                    const auto rows_of = [&](uint32_t k_block) {
                        const uint32_t k_length = k_rows(k_block);
                        if (!spread) {
                            return km::Partition{0, k_length};
                        }
                        return km::partition(k_length, broadcast.rank(), column_ranks);
                    };
                    // Those rows read into `into` (its first row there), a row's tiles
                    // at consecutive indices of this island's part of W
                    const auto read_rows = [&](uint32_t k_block, uint32_t into) {
                        const km::Partition rows = rows_of(k_block);
                        for (uint32_t row = 0; row < rows.len; ++row) {
                            const uint32_t index = (
                                w_here.index(
                                    (
                                        mine.first + k_block * k_block_tiles +
                                        rows.start + row
                                    ) *
                                        n_whole +
                                    window.first + block_column
                                )
                            );
                            for (uint32_t column = 0; column < columns_read; ++column) {
                                noc_async_read_page(
                                    index + column,
                                    w_matrix,
                                    into + (row * block_width + column) * kBf16Bytes
                                );
                            }
                        }
                    };
                    bool ahead = false;  // a receiver's rows of this step already read
                    for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                        LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                        const uint32_t marker = (
                            (command.sequence & 0xFFFFu) << 16 | ((step + 1) & 0xFFFFu)
                        );
                        if (!broadcast.sender()) {
                            if (spread) {
                                // its rows into the sender's slot once reserved, then
                                // the next block's rows read ahead
                                if (!ahead) {
                                    read_rows(k_block, slice_base);
                                }
                                const km::Partition rows = rows_of(k_block);
                                if (
                                    !marked(writer, command, kPortReady, marker) ||
                                    !flushed(writer, km::kStWriter, km::kFlushReads)
                                ) {
                                    return false;
                                }
                                if (rows.len != 0) {
                                    noc_async_write(
                                        slice_base,
                                        writer.core_noc(
                                            cast_arg(
                                                offsetof(km::BroadcastArgs, sender)
                                            ),
                                            column_cb.fifo_limit - column_cb.fifo_size +
                                                step % broadcast.window() *
                                                    block_bytes +
                                                rows.start * block_width * kBf16Bytes
                                        ),
                                        rows.len * block_width * kBf16Bytes
                                    );
                                }
                                if (!landed(writer, command)) {
                                    return false;
                                }
                                noc_semaphore_inc(
                                    writer.core_noc(
                                        cast_arg(offsetof(km::BroadcastArgs, sender)),
                                        port_base + kPortSlices * KM_PORT_STRIDE +
                                            4 * KM_PORT_COUNT
                                    ),
                                    1
                                );
                                ahead = k_block + 1 < k_blocks;
                                if (ahead) {
                                    read_rows(k_block + 1, slice_base);
                                }
                                LINEAR_ACCOUNT_DO(lap(account[kAcWRead - 16], since));
                            }
                            if (!broadcast.arrive(step)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcWCast - 16], since));
                        } else {
                            const uint32_t base = broadcast.reserve(step);
                            if (base == 0) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcWFree - 16], since));
                            const km::Partition rows = rows_of(k_block);
                            if (spread) {  // the receivers may fill their rows now
                                // the marker from the same word (and 16-byte offset)
                                // as it lands at: this sender's own ready port
                                volatile tt_l1_ptr uint32_t* words = (
                                    km::port_words(writer, kPortReady)
                                );
                                words[KM_PORT_COUNT] = marker;
                                noc_semaphore_set_multicast(
                                    port_base + kPortReady * KM_PORT_STRIDE +
                                        4 * KM_PORT_COUNT,
                                    get_noc_multicast_addr(
                                        mc_start & 0xFFFFu,
                                        mc_start >> 16,
                                        mc_end & 0xFFFFu,
                                        mc_end >> 16,
                                        port_base + kPortReady * KM_PORT_STRIDE +
                                            4 * KM_PORT_COUNT
                                    ),
                                    column_ranks - 1
                                );
                            }
                            read_rows(
                                k_block, base + rows.start * block_width * kBf16Bytes
                            );
                            if (
                                spread &&
                                !counted(writer, command, kPortSlices, column_ranks - 1)
                            ) {
                                return false;
                            }
                            if (!flushed(writer, km::kStWriter, km::kFlushReads)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcWRead - 16], since));
                            LINEAR_ACCOUNT_DO(
                                account[kAcWTiles - 16] += k_rows(k_block) *
                                    columns_read
                            );
                            if (!broadcast.cast(step)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcWCast - 16], since));
                        }
                        if (k_block == 0 && pending) {
                            if (!made(pending_block, pending_row, pending_column)) {
                                return false;
                            }
                            pending = false;
                        }
                    }
                    pending = true;
                    pending_block = block;
                    pending_row = block_row;
                    pending_column = block_column;
                }
                if (pending && !made(pending_block, pending_row, pending_column)) {
                    return false;
                }
                if (adding) {
                    // The partial in S, the other island's brought here; the reader
                    // is told, and compute's sums become Y.
                    const uint32_t tiles = blocks_high * blocks_wide * block_tiles;
                    const uint32_t rank = (
                        km::group_of(command, writer.me()).local().rank
                    );
                    km::Gather posted = {};
                    // the column broadcast ends first: compute pops its padding before
                    // the addition
                    finished = true;
                    if (
                        !broadcast.finish() || !landed(writer, command) ||
                        !km::gather_pages(
                            writer,
                            command,
                            s,
                            sent.page(add_set(parity, here), 0),
                            tiles,
                            island_at(here, 1),
                            posted,
                            km::keeper(rank, kPortAdd)
                        )
                    ) {
                        return false;
                    }
                    stamps[kStampPut] = km::wall();
                    if (!km::gathered(writer, command, kPortAdd, tiles)) {
                        return false;
                    }
                    stamps[kStampSettled] = km::wall();
                    stamps[kStampUp] = stamps[kStampSettled];
                    stamps[kStampDown] = stamps[kStampSettled];
                    if (!tell(writer, command, writer.me(), kPortGo)) {
                        return false;
                    }
                    stamps[kStampSynced] = km::wall();
                    for (
                        uint32_t block = 0; block < blocks_high * blocks_wide; ++block
                    ) {
                        if (!put(
                            block,
                            first_row + block / blocks_wide * block_height,
                            first_column + block % blocks_wide * block_width
                        )) {
                            return false;
                        }
                    }
                    if (!km::gather_wait(writer, command, posted)) {
                        return false;
                    }
                }
            }
            if (!finished && !broadcast.finish()) {
                return false;
            }
        }
        stamps[kStampEnd] = km::wall();
        parity ^= 1u;
        LINEAR_ACCOUNT_DO(for (uint32_t word = kAcWFree; word <= kAcYWrite; ++word) {
            stamps[word] = account[word - kAcWFree];
        });
        if (command.a6 == 0) {
            return true;
        }
        const uint32_t rank = km::group_of(command, writer.me()).local().rank;
        noc_async_write_page(
            rank,
            TensorAccessor(t_args, holding(command.a6, km::island(), 1).address),
            get_write_ptr(kCbStamp)
        );
        return flushed(writer, km::kStWriter, km::kFlushSent);
    });
}
#endif  // LINEAR_SPLIT
