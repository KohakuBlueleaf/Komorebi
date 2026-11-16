// The spin program's reader: SPIN's units, each a wait of the cycles its cost line
// names (the data decides the cost), then a line to the outs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/core/km_program.hpp"
#include "../../programs.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SPIN) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, reader.me() - km::first(command), km::count(command))
    );
    using Costs = km::Tensor<0>;
    const Costs costs(command.a2);
    const km::Tensor<Costs::next_compile_time_args_offset()> outs(command.a3);
    if (!reader.reserve(TP_CB_LINE, 1)) {
        return;
    }
    const uint32_t buffer = get_write_ptr(TP_CB_LINE);
    volatile tt_l1_ptr uint32_t* line = km::l1(buffer);
    for (uint32_t index = 0; index < partition.len; index++) {
        const uint32_t unit = command.a0 + partition.start + index;
        if (!km::read_pages(reader, command, costs, unit, 1, buffer)) {
            return;
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        const uint32_t cost = line[0];
        const uint32_t start = km::wall();
        while (km::wall() - start < cost) {
            ;
        }
        line[0] = TP_MARK(unit, cost, command.a4);
        line[1] = unit;
        line[2] = command.a4;
        if (!km::write_pages(reader, command, outs, unit, 1, buffer)) {
            return;
        }
        noc_async_write_barrier();
    }
}
