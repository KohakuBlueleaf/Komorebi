// Linear reader (NoC0 by default): X's row broadcast over this island's K for its own
// window of columns (LINEAR_ADD: then both partials of each subblock, to compute).
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
            uint32_t step = 0;
            for (
                uint32_t block = 0; width != 0 && block < blocks_high * blocks_wide;
                ++block
            ) {
                const uint32_t block_row = (
                    first_row + block / blocks_wide * block_height
                );
                for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                    LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                    if (!broadcast.sender()) {
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
                        const uint32_t k_start = k_block * k_block_tiles;
                        const uint32_t k_length = (
                            mine.count - k_start < k_block_tiles
                                ? mine.count - k_start
                                : k_block_tiles
                        );
                        const uint32_t rows = (
                            block_row >= made_end
                                ? 0
                                : made_end - block_row < block_height
                                    ? made_end - block_row
                                    : block_height
                        );
                        for (uint32_t tile_row = 0; tile_row < rows; ++tile_row) {
                            // a row of X: this island's K, its part's pages
                            const uint32_t page = (
                                (block_row + tile_row) * k_whole + mine.first + k_start
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
