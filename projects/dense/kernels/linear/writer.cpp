// Linear writer (NoC1 by default): W's column broadcast a window of columns; other
// islands' tiles put in chunks through the link cores' rings, one sync, its own Y.
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

    // Every island's partial sums are in its DRAM once this returns on every core:
    // up the island's tree, its leader to the other leaders, down the tree.
    bool everywhere(
        const km::Writer& writer,
        const km_command_t& command,
        volatile tt_l1_ptr uint32_t* stamps
    ) {
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
        stamps[kStampUp] = km::wall();
        if (rank != 0) {
            if (!tell(writer, command, km::core_of(local, (rank - 1) / 2), kPortUp)) {
                return false;
            }
            if (!counted(writer, command, kPortDown, 1)) {
                return false;
            }
            stamps[kStampDown] = km::wall();
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
            stamps[kStampDown] = km::wall();
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
    // the tiles put: chunks from a buffer of two, its place kept across commands (a
    // put never crosses the buffer's end)
    const uint32_t chunk = put_chunk(plan(kPlSh) * plan(kPlSw));
    uint32_t ring_at = 0;
    uint32_t parity = 0;  // LINEAR_ADD: the command's set of S
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return true;
        }
        volatile tt_l1_ptr uint32_t* stamps = km::l1(get_write_ptr(kCbStamp));
        stamps[kStampStart] = km::wall();
        stamps[kStampPut] = 0;  // 0: no sync this command
        stamps[kStampSynced] = 0;
        stamps[kStampSettled] = 0;
        stamps[kStampUp] = 0;
        stamps[kStampDown] = 0;
        LINEAR_ACCOUNT_DO(uint32_t account[kAcYWrite - kAcWFree + 1] = {});
        for (uint32_t word = kStampFed; word <= kStampPieces; ++word) {
            stamps[word] = 0;
        }
        const uint32_t w = command.a1, y = command.a2, s = command.a3;
        const uint32_t k_whole = command.a4, n_whole = command.a5;
        const uint32_t m_tiles = plan(kPlMt), here = km::island();
        const uint32_t islands = km::islands();
        invalidate_l1_cache();  // the tensor table
        const Span mine = k_part(w, here, k_whole, n_whole);
        const uint32_t x = command.a0;
        bool sync = LINEAR_EXCHANGES && exchanging(x, w, y, m_tiles, k_whole, n_whole);
        // the rows this island makes: the plan's rows start at its first
        const Span made = made_rows(x, here, m_tiles, k_whole);
        const auto window_of = [&](uint32_t to) {
            return window(x, w, y, here, to, m_tiles, k_whole, n_whole);
        };
        const Holding y_here = holding(y, here, m_tiles * n_whole);
        const auto y_matrix = TensorAccessor(y_args, y_here.address);
        // the input-dim-sharded form's addition here: two islands, Y a copy
        const bool adding = LINEAR_ADDS && islands == 2 && copied(y);
        const auto s_matrix = TensorAccessor(s_args, holding(s, here, 1).address);
        // this core's pages put and not yet counted (their tiles already freed)
        uint32_t outstanding = 0, holder = here;
        km::PutRun run;
        const uint32_t block_height = plan(kPlBm), block_width = plan(kPlBn);
        const uint32_t block_tiles = block_height * block_width;
        const auto settle = [&] {
            if (outstanding == 0) {
                return true;
            }
            if (!km::put_wait(writer, command, outstanding, holder)) {
                return false;
            }
            outstanding = 0;
            return true;
        };
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
            uint32_t step = 0;
            bool finished = false;  // the column broadcast (LINEAR_ADD: before the sum)
            for (uint32_t turn = 1; turn <= islands; ++turn) {
                const uint32_t to = island_at(here, turn);
                const Span window = window_of(to);
                if (window.count == 0 || (!LINEAR_EXCHANGES && to != here)) {
                    continue;
                }
                const bool own = !LINEAR_EXCHANGES || to == here;
                if (own && sync) {  // every put counted, then every island's
                    stamps[kStampPut] = km::wall();
                    if (!settle()) {
                        return false;
                    }
                    stamps[kStampSettled] = km::wall();
                    if (!everywhere(writer, command, stamps)) {
                        return false;
                    }
                    stamps[kStampSynced] = km::wall();
                    sync = false;
                }
                const uint32_t column_end = (
                    first_column + plan(kPlPcN) < window.count
                        ? first_column + plan(kPlPcN)
                        : window.count
                );
                const uint32_t s_address = holding(s, to, 1).address;
                const uint32_t s_set = sent_set(here, to);

                // Another island's tiles handed .. `upto` - 1 of this core's stream
                // (its blocks in order) put to S in chunks; `rest`: a last shorter one.
                uint32_t handed = 0;
                const auto send = [&](uint32_t upto, bool rest) {
                    while (handed < upto && (upto - handed >= chunk || rest)) {
                        uint32_t count = upto - handed < chunk ? upto - handed : chunk;
                        if (count > 2 * chunk - ring_at) {
                            count = 2 * chunk - ring_at;  // to the buffer's end
                        }
                        uint32_t clock = km::wall();
                        const auto lap = [&](uint32_t word) {
                            const uint32_t now = km::wall();
                            stamps[word] += now - clock;
                            clock = now;
                        };
                        if (!writer.wait_front(kCbSend, count)) {
                            return false;
                        }
                        lap(kStampFed);
                        // a run of `count` pages, its bytes into the rings at once
                        if (!km::put_pages(
                            writer,
                            command,
                            to,
                            s_address,
                            kBf16Bytes,
                            sent.page(s_set, handed),
                            count,
                            run
                        )) {
                            return false;
                        }
                        lap(kStampLine);
                        if (!km::put_piece(
                            writer, command, run, get_read_ptr(kCbSend), count
                        )) {
                            return false;
                        }
                        lap(kStampPiece);
                        stamps[kStampTurn] += run.turn;
                        stamps[kStampWrite] += run.write;
                        stamps[kStampPieces] += run.piece;
                        outstanding += count - run.waited;  // no rings: waited there
                        holder = to;
                        cb_pop_front(kCbSend, count);
                        handed += count;
                        ring_at = (ring_at + count) % (2 * chunk);
                    }
                    return true;
                };
                // Tiles handed .. `upto` - 1 freed unput (a block outside the window).
                const auto drop = [&](uint32_t upto) {
                    while (handed < upto) {
                        uint32_t count = upto - handed < chunk ? upto - handed : chunk;
                        if (count > 2 * chunk - ring_at) {
                            count = 2 * chunk - ring_at;
                        }
                        if (!writer.wait_front(kCbSend, count)) {
                            return false;
                        }
                        cb_pop_front(kCbSend, count);
                        handed += count;
                        ring_at = (ring_at + count) % (2 * chunk);
                    }
                    return true;
                };

                // Output block `block` at (block_row, block_column) made: its own
                // window's Y written by subblock; another island's tiles to S.
                const auto put = [&](
                    uint32_t block, uint32_t block_row, uint32_t block_column
                ) {
                    if (!own) {
                        const uint32_t made = (block + 1) * block_tiles;
                        const bool last = block + 1 == blocks_high * blocks_wide;
                        if (block_row < row_end && block_column < column_end) {
                            return send(made, last);
                        }
                        return send(made - block_tiles, true) && drop(made);
                    }
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
                    for (uint32_t k_block = 0; k_block < k_blocks; ++k_block, ++step) {
                        LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
                        if (!broadcast.sender()) {
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
                            const uint32_t k_start = k_block * k_block_tiles;
                            const uint32_t k_length = (
                                mine.count - k_start < k_block_tiles
                                    ? mine.count - k_start
                                    : k_block_tiles
                            );
                            for (uint32_t k_tile = 0; k_tile < k_length; ++k_tile) {
                                // W's rows this island holds, read here: a row's
                                // tiles at consecutive indices of its part
                                const uint32_t index = (
                                    w_here.index(
                                        (mine.first + k_start + k_tile) * n_whole +
                                        window.first + block_column
                                    )
                                );
                                for (
                                    uint32_t column = 0; column < columns_read; ++column
                                ) {
                                    noc_async_read_page(
                                        index + column,
                                        w_matrix,
                                        base +
                                            (k_tile * block_width + column) * kBf16Bytes
                                    );
                                }
                            }
                            if (!flushed(writer, km::kStWriter, km::kFlushReads)) {
                                return false;
                            }
                            LINEAR_ACCOUNT_DO(lap(account[kAcWRead - 16], since));
                            LINEAR_ACCOUNT_DO(
                                account[kAcWTiles - 16] += k_length * columns_read
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
                if (!adding) {
                    continue;
                }
                // The partial in S goes to the other island's S, its partial of these
                // blocks comes here; the reader is told, and compute's sums become Y.
                const uint32_t tiles = blocks_high * blocks_wide * block_tiles;
                const uint32_t rank = km::group_of(command, writer.me()).local().rank;
                km::Gather posted = {};
                // the column broadcast ends first: compute pops its padding before the
                // addition
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
                for (uint32_t block = 0; block < blocks_high * blocks_wide; ++block) {
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
            if (!settle() || (!finished && !broadcast.finish())) {
                return false;
            }
        }
        if (sync) {
            stamps[kStampPut] = km::wall();
            stamps[kStampSettled] = stamps[kStampPut];
            if (!everywhere(writer, command, stamps)) {
                return false;
            }
            stamps[kStampSynced] = km::wall();
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
