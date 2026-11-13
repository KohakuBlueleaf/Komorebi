// The reader kernel: the leader of the row reads the values and, for each one, casts
// what the row does next; every core does it and writes its total.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto values_arguments = TensorAccessorArgs<0>();
    constexpr uint32_t totals_offset = values_arguments.next_compile_time_args_offset();
    constexpr auto totals_arguments = TensorAccessorArgs<totals_offset>();
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerFollow) {
            return;
        }
        km::Broadcast row(reader, command, 0);
        if (!row.ok() || !row.included()) {
            return;
        }
        const uint32_t count = command.a1;
        const auto values = (
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kValues))
        );
        if (row.sender()) {
            const auto source = TensorAccessor(values_arguments, command.a0);
            noc_async_read_page(0, source, get_write_ptr(kValues));
            const uint32_t code = reader.poll(
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                reader.args().wait_limit
            );
            if (code != km::kGaveNone) {
                return;
            }
        }
        uint32_t total = 1;
        for (uint32_t cast = 0; cast < count; cast++) {
            if (row.sender()) {
                const uint32_t choice = values[cast] % 2 == 0 ? kDouble : kAddThree;
                if (!row.cast(cast, choice)) {
                    return;
                }
            } else if (!row.arrive(cast)) {
                return;
            }
            total = row.word() == kDouble ? total * 2 : total + 3;
        }
        if (!row.finish()) {
            return;
        }
        // --8<-- [end:run]
        const auto out = (
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(kTotal))
        );
        out[0] = total;
        const auto totals = TensorAccessor(totals_arguments, command.a2);
        const uint32_t rank = reader.me() - km::first(command);
        noc_async_write_page(rank, totals, get_write_ptr(kTotal));
        reader.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            reader.args().wait_limit
        );
    });
}
