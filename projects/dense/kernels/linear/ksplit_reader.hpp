// Linear reader, the K-split plan (NoC0, mover A): its chunks of this rank's K (every
// other one; the rest on the writer), then the parts this rank sums (owner or tree).
#include "komorebi/device/movement/km_port.hpp"
#include "ksplit.hpp"

void kernel_main() {
    using X = km::Tensor<0>;
    using W = km::Tensor<X::next_compile_time_args_offset()>;
    km::Reader reader;
    const auto plan = [&](uint32_t word) {
        return reader.arg(word);
    };
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLinear) {
            return;
        }
        LINEAR_ACCOUNT_DO(uint32_t since = km::wall());
        LINEAR_ACCOUNT_DO(uint32_t moved = 0);
        const KPlace at = k_place(command, reader.me(), plan, km::island());
        if (!at.any) {
            return;
        }
        if (at.k_count == 0) {  // a rank of a group with no K: its sum never written
            reader.gave_up(
                km::kStReader,
                km::kGaveBadItem,
                km::kWhereData,
                km::slot(command),
                command.sequence
            );
            return;
        }
        if (!move_k_chunks(reader, command, plan, at, X(command.a0), W(command.a1))) {
            return;
        }
        LINEAR_ACCOUNT_DO(lap(moved, since));
        uint32_t index = 0;
        const km::Group group = k_group(command, reader.me(), plan(kKsS), index);
        km::reduce_arrive(
            reader,
            command,
            {kCbKArrive, plan(kKsArrive), plan(kKsOh) * plan(kKsOw)},
            group,
            plan(kKsForm)
        );
        LINEAR_ACCOUNT_DO({
            volatile tt_l1_ptr uint32_t* stamps = km::l1(get_write_ptr(kCbStamp));
            stamps[kAcKMoveA] = moved;
            stamps[kAcKReaderReduce] = km::wall() - since;
        });
    });
}
