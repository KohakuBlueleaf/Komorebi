// Linear reader (NoC0 by default): X's row broadcast over this island's K for its own
// window of columns (LINEAR_ADD: then both partials of each subblock, to compute).
#include <stddef.h>
#include <stdint.h>

#include "../dense.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "komorebi/device/km_port.hpp"
#include "linear.h"

#if defined(LINEAR_SPLIT)  // the K-split plan (linear.h): its own body
#include "ksplit_reader.hpp"
#else

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
    uint32_t parity = 0;  // LINEAR_ADD: the command's set of S
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
        // the rows this island makes: the plan's rows start at its first
        const Span made = made_rows(x, here, m_tiles, k_whole);
        const bool included = core(kCoreRank) != 0;
        const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
        const uint32_t k_block_tiles = plan(kPlKb);
        const uint32_t k_blocks = blocks_of(mine.count, k_block_tiles);
        const uint32_t blocks_high = plan(kPlNbh), blocks_wide = plan(kPlNbw);
        const uint32_t first_row = made.first + core(kCoreM0);
        const uint32_t made_end = made.first + made.count;
        const uint32_t subblock_tiles = plan(kPlSh) * plan(kPlSw);
        const uint32_t block_tiles = block_height * block_width;
        const Sent sent = {core(kCoreSetPages), core(kCoreSent)};
        // this island's own columns of the rows it makes (none: nothing to make)
        const uint32_t width = (
            window(x, w, y, here, here, m_tiles, k_whole, n_whole).count
        );
        LINEAR_ACCOUNT_DO(uint32_t account[kAcReaderEnd - kAcXFree + 1] = {});
        const Holding x_here = holding(x, here, m_tiles * k_whole);
        const auto x_matrix = TensorAccessor(x_args, x_here.address);
        const auto s_matrix = TensorAccessor(s_args, holding(s, here, 1).address);
        if (included && mine.count != 0) {
            km::Broadcast broadcast(reader, command, kArgBroadcast);
#if defined(LINEAR_VALID_SENT)
            broadcast.valid_when_sent();
#endif
            if (!broadcast.ok()) {
                return;
            }
            // LINEAR_XSPREAD: the row broadcast's ranks (sender 0) when a receiver's
            // share fits its slice; the sender's slots, the multicast area
            const auto cast_arg = [&](uint32_t offset) {
                return reader.arg(kArgBroadcast + offset / 4);
            };
            const uint32_t row_ranks = (
                cast_arg(offsetof(km::BroadcastArgs, receivers)) + 1
            );
            const uint32_t mc_start = cast_arg(offsetof(km::BroadcastArgs, mc_start));
            const uint32_t mc_end = cast_arg(offsetof(km::BroadcastArgs, mc_end));
            const uint32_t sender = cast_arg(offsetof(km::BroadcastArgs, sender));
            const LocalCBInterface& slice_cb = get_local_cb_interface(kCbXSlice);
            const LocalCBInterface& row_cb = get_local_cb_interface(kCbRow);
            const uint32_t block_bytes = broadcast.pages() * row_cb.fifo_page_size;
            const bool spread = (
                LINEAR_XSPREADS && broadcast.included() && row_ranks > 1 &&
                blocks_of(block_height * k_block_tiles, row_ranks) * kBf16Bytes <=
                    slice_cb.fifo_size
            );
            const uint32_t slice_base = slice_cb.fifo_limit - slice_cb.fifo_size;
            const uint32_t port_base = reader.args().control + KM_CONTROL_PORT;
            uint32_t step = 0;
            for (
                uint32_t block = 0; width != 0 && block < blocks_high * blocks_wide;
                ++block
            ) {
                const uint32_t block_row = (
                    first_row + block / blocks_wide * block_height
                );
                const uint32_t rows = (
                    block_row >= made_end
                        ? 0
                        : made_end - block_row < block_height
                            ? made_end - block_row
                            : block_height
                );
                const auto k_length_of = [&](uint32_t k_block) {
                    const uint32_t k_start = k_block * k_block_tiles;
                    return (
                        mine.count - k_start < k_block_tiles
                            ? mine.count - k_start
                            : k_block_tiles
                    );
                };
                // The block's tiles (rows x K block, row by row) this rank reads: all
                // on the sender, unless spread
                const auto share_of = [&](uint32_t k_block) {
                    const uint32_t tiles = rows * k_length_of(k_block);
                    if (!spread) {
                        return km::Partition{0, tiles};
                    }
                    return km::partition(tiles, broadcast.rank(), row_ranks);
                };
                // Those tiles' runs along a row of the block: at(row, first tile of the
                // run in the share, k, tiles) for each run. False: X is not held here.
                const auto runs = [&](uint32_t k_block, auto&& at) {
                    const uint32_t k_length = k_length_of(k_block);
                    const km::Partition share = share_of(k_block);
                    for (uint32_t t = share.start; t < share.start + share.len;) {
                        const uint32_t tile_row = t / k_length, k = t % k_length;
                        const uint32_t end = (tile_row + 1) * k_length;
                        const uint32_t run = (
                            share.start + share.len < end
                                ? share.start + share.len - t
                                : end - t
                        );
                        if (!at(tile_row, t - share.start, k, run)) {
                            return false;
                        }
                        t += run;
                    }
                    return true;
                };
                // Those tiles read: a run's into `into(tile_row, first, k)`
                const auto read_share = [&](uint32_t k_block, auto&& into) {
                    const uint32_t k_start = k_block * k_block_tiles;
                    return runs(
                        k_block,
                        [&](
                            uint32_t tile_row, uint32_t first, uint32_t k, uint32_t run
                        ) {
                            // a row of X: this island's K, its part's pages
                            const uint32_t page = (
                                (block_row + tile_row) * k_whole + mine.first +
                                k_start + k
                            );
                            if (!x_here.holds(page, run)) {
                                reader.gave_up(  // X is not split as W is
                                    km::kStReader,
                                    km::kGaveBadItem,
                                    km::kWhereData,
                                    km::slot(command),
                                    command.sequence
                                );
                                return false;
                            }
                            const uint32_t index = x_here.index(page);
                            const uint32_t at = into(tile_row, first, k);
                            for (uint32_t tile = 0; tile < run; ++tile) {
                                noc_async_read_page(
                                    index + tile, x_matrix, at + tile * kBf16Bytes
                                );
                            }
                            return true;
                        }
                    );
                };
                const auto in_slice = [&](uint32_t, uint32_t first, uint32_t) {
                    return slice_base + first * kBf16Bytes;
                };
                bool ahead = false;  // a receiver's share of this step already read
                for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                    LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                    const uint32_t marker = (
                        (command.sequence & 0xFFFFu) << 16 | ((step + 1) & 0xFFFFu)
                    );
                    if (!broadcast.sender()) {
                        if (spread) {
                            // its share into the sender's slot once reserved, then the
                            // next block's share read ahead
                            if (!ahead && !read_share(k_block, in_slice)) {
                                return;
                            }
                            if (!marked(
                                reader, km::kStReader, command, kPortXReady, marker
                            ) ||
                                !flushed(reader, km::kStReader, km::kFlushReads)) {
                                return;
                            }
                            const uint32_t slot = (
                                row_cb.fifo_limit - row_cb.fifo_size +
                                step % broadcast.window() * block_bytes
                            );
                            runs(
                                k_block,
                                [&](
                                    uint32_t tile_row,
                                    uint32_t first,
                                    uint32_t k,
                                    uint32_t run
                                ) {
                                    noc_async_write(
                                        slice_base + first * kBf16Bytes,
                                        reader.core_noc(
                                            sender,
                                            slot +
                                                (tile_row * k_block_tiles + k) *
                                                    kBf16Bytes
                                        ),
                                        run * kBf16Bytes
                                    );
                                    return true;
                                }
                            );
                            if (!landed(reader, km::kStReader, command)) {
                                return;
                            }
                            noc_semaphore_inc(
                                reader.core_noc(
                                    sender,
                                    port_base + kPortXSlices * KM_PORT_STRIDE +
                                        4 * KM_PORT_COUNT
                                ),
                                1
                            );
                            ahead = k_block + 1 < k_blocks;
                            if (ahead && !read_share(k_block + 1, in_slice)) {
                                return;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcXRead - kAcXFree], since));
                        }
                        if (!broadcast.arrive(step)) {
                            return;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[kAcXCast - kAcXFree], since));
                    } else {
                        const uint32_t base = broadcast.reserve(step);
                        if (base == 0) {
                            return;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[0], since));
                        if (spread) {  // the receivers may fill their shares now
                            // the marker from the same word (and 16-byte offset) as
                            // it lands at: this sender's own ready port
                            volatile tt_l1_ptr uint32_t* words = (
                                km::port_words(reader, kPortXReady)
                            );
                            words[KM_PORT_COUNT] = marker;
                            noc_semaphore_set_multicast(
                                port_base + kPortXReady * KM_PORT_STRIDE +
                                    4 * KM_PORT_COUNT,
                                get_noc_multicast_addr(
                                    mc_start & 0xFFFFu,
                                    mc_start >> 16,
                                    mc_end & 0xFFFFu,
                                    mc_end >> 16,
                                    port_base + kPortXReady * KM_PORT_STRIDE +
                                        4 * KM_PORT_COUNT
                                ),
                                row_ranks - 1
                            );
                        }
                        if (!read_share(
                            k_block, [&](uint32_t tile_row, uint32_t, uint32_t k) {
                                return (
                                    base + (tile_row * k_block_tiles + k) * kBf16Bytes
                                );
                            }
                        )) {
                            return;
                        }
                        if (
                            spread &&
                            !counted(
                                reader,
                                km::kStReader,
                                command,
                                kPortXSlices,
                                row_ranks - 1
                            )
                        ) {
                            return;
                        }
                        if (!flushed(reader, km::kStReader, km::kFlushReads)) {
                            return;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[kAcXRead - kAcXFree], since));
                        if (!broadcast.cast(step)) {
                            return;
                        }
                        LINEAR_ACCOUNT_DO(lap(account[kAcXCast - kAcXFree], since));
                    }
                }
            }
            if (!broadcast.finish()) {
                return;
            }
            // LINEAR_ADD (two islands, Y a copy): told both partials are in S, each
            // subblock of this island's then of the other's to compute
            if (LINEAR_ADDS && islands == 2 && copied(y)) {
                if (!km::port_wait(reader, command, kPortGo, 1)) {
                    return;
                }
                const uint32_t sets[2] = {
                    add_set(parity, here), add_set(parity, island_at(here, 1))
                };
                for (
                    uint32_t tile = 0; tile < blocks_high * blocks_wide * block_tiles;
                    tile += subblock_tiles
                ) {
                    for (const uint32_t set : sets) {
                        if (!reader.reserve(kCbSent, subblock_tiles)) {
                            return;
                        }
                        const uint32_t at = get_write_ptr(kCbSent);
                        for (uint32_t k = 0; k < subblock_tiles; ++k) {
                            noc_async_read_page(
                                sent.page(set, tile + k), s_matrix, at + k * kBf16Bytes
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
        parity ^= 1u;
        LINEAR_ACCOUNT_DO({
            account[kAcReaderEnd - kAcXFree] = km::wall();
            volatile tt_l1_ptr uint32_t* stamps = km::l1(get_write_ptr(kCbStamp));
            for (uint32_t word = kAcXFree; word <= kAcReaderEnd; ++word) {
                stamps[word] = account[word - kAcXFree];
            }
        });
    });
}
#endif  // LINEAR_SPLIT
