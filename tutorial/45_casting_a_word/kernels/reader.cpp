// The reader kernel: the leader of the row reads the values and, for each one, casts
// what the row does next; every core does it and writes its total.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/movement/km_broadcast.hpp"
#include "handlers.h"

void kernel_main() {
    using Values = km::Tensor<0>;
    using Totals = km::Tensor<Values::next_compile_time_args_offset()>;
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
            const Values source(command.a0);
            if (!km::read_pages(
                reader, command, source, 0, 1, get_write_ptr(kValues)
            )) {
                return;
            }
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
        const Totals totals(command.a2);
        const uint32_t rank = reader.me() - km::first(command);
        if (!km::write_pages(reader, command, totals, rank, 1, get_write_ptr(kTotal))) {
            return;
        }
        reader.poll(
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            reader.args().wait_limit
        );
    });
}
