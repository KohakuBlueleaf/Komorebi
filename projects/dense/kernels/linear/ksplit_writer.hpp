// Linear writer, the K-split plan (NoC1, mover B): its K chunks; then its part (fp32)
// sent on, or by the group's first rank the group's sum (bf16) written to Y.
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_port.hpp"
#include "ksplit.hpp"

void kernel_main() {
    using Y = km::Tensor<0>;
    using X = km::Tensor<Y::next_compile_time_args_offset()>;
    using W = km::Tensor<X::next_compile_time_args_offset()>;
    constexpr auto t_args = TensorAccessorArgs<W::next_compile_time_args_offset()>();
    using S = km::Tensor<t_args.next_compile_time_args_offset()>;
    km::Writer writer;
    uint32_t parity = 0;  // LINEAR_ADD: the command's set of S
    const auto plan = [&](uint32_t word) {
        return writer.arg(word);
    };
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return true;
        }
        volatile tt_l1_ptr uint32_t* stamps = km::l1(get_write_ptr(kCbStamp));
        stamps[kStampStart] = km::wall();
        for (uint32_t word = kStampPut; word < kStampBytes / 4; ++word) {
            stamps[word] = 0;  // no puts, no sync (the reader's account words later)
        }
        LINEAR_ACCOUNT_DO(uint32_t since = stamps[kStampStart]);
        LINEAR_ACCOUNT_DO(uint32_t account[kAcKWrite - kAcKMoveB + 1] = {});
        const KPlace at = k_place(command, writer.me(), plan, km::island());
        const uint32_t block_height = plan(kKsOh), block_width = plan(kKsOw);
        const uint32_t block_tiles = block_height * block_width;
        if (at.any) {
            const X x_matrix(command.a0);
            const W w_matrix(command.a1);
            if (!move_k_chunks(writer, command, plan, at, x_matrix, w_matrix)) {
                return false;
            }
            LINEAR_ACCOUNT_DO(lap(account[kAcKMoveB - kAcKMoveB], since));
            uint32_t index = 0;
            const km::Group group = k_group(command, writer.me(), plan(kKsS), index);
            const km::Arrival arrival = {kCbKArrive, plan(kKsArrive), block_tiles};
            if (group.rank != 0) {
                LINEAR_ACCOUNT_DO({  // the part to send in hand (emit waits again)
                    if (!writer.wait_front(kCbKSum, block_tiles, km::kBehindPort)) {
                        return false;
                    }
                    stamps[kAcKHeld] = km::wall();
                });
                if (!km::reduce_emit(
                    writer,
                    command,
                    arrival,
                    group,
                    plan(kKsForm),
                    kCbKSum,
                    block_tiles
                )) {
                    return false;
                }
                LINEAR_ACCOUNT_DO(lap(account[kAcKReduce - kAcKMoveB], since));
            } else {
                if (!km::reduce_result(writer, kCbKOut, block_tiles)) {
                    return false;
                }
                LINEAR_ACCOUNT_DO(lap(account[kAcKReduce - kAcKMoveB], since));
                const uint32_t n_whole = command.a5;
                const Y y_matrix(command.a2);
                // LINEAR_ADD (two islands, Y a copy): the island's sum is its partial
                if (LINEAR_ADDS && km::islands() == 2 && copied(command.a2)) {
                    const S s_matrix(command.a3);
                    const uint32_t here = km::island(), far = island_at(here, 1);
                    const uint32_t set_pages = plan(kKsSetPages);
                    const uint32_t mine = add_set(parity, here) * set_pages;
                    const uint32_t theirs = add_set(parity, far) * set_pages;
                    const uint32_t offset = index * block_tiles;
                    const uint32_t rank = (
                        km::group_of(command, writer.me()).local().rank
                    );
                    km::Gather posted = {};
                    if (!km::write_pages(
                        writer,
                        command,
                        s_matrix,
                        mine + offset,
                        block_tiles,
                        get_read_ptr(kCbKOut)
                    )) {
                        return false;
                    }
                    const uint32_t code = writer.poll_at(
                        km::kWhereData,
                        [] {
                            return ncrisc_noc_nonposted_writes_flushed(noc_index);
                        },
                        writer.args().wait_limit
                    );
                    if (code != km::kGaveNone) {
                        writer.gave_up(
                            km::kStWriter,
                            code,
                            km::kWhereData,
                            km::slot(command),
                            command.sequence
                        );
                        return false;
                    }
                    cb_pop_front(kCbKOut, block_tiles);
                    if (!km::gather_pages(
                        writer,
                        command,
                        command.a3,
                        mine + offset,
                        block_tiles,
                        far,
                        posted,
                        km::keeper(rank, kPortAdd)
                    )) {
                        return false;
                    }
                    stamps[kStampPut] = km::wall();
                    if (!km::gathered(writer, command, kPortAdd, block_tiles)) {
                        return false;
                    }
                    stamps[kStampSettled] = km::wall();
                    stamps[kStampUp] = stamps[kStampSettled];
                    stamps[kStampDown] = stamps[kStampSettled];
                    writer.produce(kCbKAdd);  // empty here: compute took the last
                    cb_reserve_back(kCbKAdd, 2 * block_tiles);
                    const uint32_t both = get_write_ptr(kCbKAdd);
                    if (!km::read_pages(
                        writer, command, s_matrix, mine + offset, block_tiles, both
                    ) ||
                        !km::read_pages(
                            writer,
                            command,
                            s_matrix,
                            theirs + offset,
                            block_tiles,
                            both + block_tiles * kBf16Bytes
                        ) ||
                        !flushed(writer, km::kStWriter, km::kFlushReads)) {
                        return false;
                    }
                    cb_push_back(kCbKAdd, 2 * block_tiles);
                    stamps[kStampSynced] = km::wall();
                    if (
                        !km::gather_wait(writer, command, posted) ||
                        !writer.wait_front(kCbKOut, block_tiles)
                    ) {
                        return false;
                    }
                }
                const uint32_t source = get_read_ptr(kCbKOut);
                for (uint32_t row = 0; row < at.rows; ++row) {
                    // a row's columns: consecutive pages of Y
                    if (!km::write_pages(
                        writer,
                        command,
                        y_matrix,
                        (at.first_row + row) * n_whole + at.first_column,
                        at.columns,
                        source + row * block_width * kBf16Bytes
                    )) {
                        return false;
                    }
                }
                if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                    return false;
                }
                cb_pop_front(kCbKOut, block_tiles);
                LINEAR_ACCOUNT_DO(lap(account[kAcKWrite - kAcKMoveB], since));
            }
        }
        // LINEAR_GATHER: the islands' columns into every copy of Y, the gather's own
        // term in the stamps (put: its start, synced: its end)
        if (LINEAR_GATHERS && km::islands() > 1 && copied(command.a2)) {
            stamps[kStampPut] = km::wall();
            if (!km::all_gather(
                writer, command, command.a2, 0, plan(kKsMt) * command.a5
            )) {
                return false;
            }
            stamps[kStampSettled] = km::wall();
            stamps[kStampUp] = stamps[kStampSettled];
            stamps[kStampDown] = stamps[kStampSettled];
            stamps[kStampSynced] = stamps[kStampSettled];
        }
        stamps[kStampEnd] = km::wall();
        parity ^= 1u;
        LINEAR_ACCOUNT_DO(for (uint32_t word = kAcKMoveB; word <= kAcKWrite; ++word) {
            stamps[word] = account[word - kAcKMoveB];
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
