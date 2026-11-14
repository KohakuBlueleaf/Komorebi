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
    km::Writer writer;
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
        stamps[kStampEnd] = km::wall();
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
