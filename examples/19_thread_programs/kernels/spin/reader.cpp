// The spin program's reader: SPIN's units, each a wait of the cycles its cost line
// names (the data decides the cost), then a line to the outs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../../programs.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SPIN) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, reader.me() - km::first(command), km::count(command))
    );
    const InterleavedAddrGen<true> costs = {
        .bank_base_address = command.a2, .page_size = TP_LINE_BYTES
    };
    const InterleavedAddrGen<true> outs = {
        .bank_base_address = command.a3, .page_size = TP_LINE_BYTES
    };
    if (!reader.reserve(TP_CB_LINE, 1)) {
        return;
    }
    const uint32_t buffer = get_write_ptr(TP_CB_LINE);
    volatile tt_l1_ptr uint32_t* line = km::l1(buffer);
    for (uint32_t index = 0; index < partition.len; index++) {
        const uint32_t unit = command.a0 + partition.start + index;
        noc_async_read(get_noc_addr(unit, costs), buffer, TP_LINE_BYTES);
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
        noc_async_write(buffer, get_noc_addr(unit, outs), TP_LINE_BYTES);
        noc_async_write_barrier();
    }
}
