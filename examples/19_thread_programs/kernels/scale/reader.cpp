// The scale program's reader: SCALE reads the core's partition of the input tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_program.hpp"
#include "../../programs.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SCALE) {
        return;
    }
    const km::Partition partition = (
        km::partition(command.a1, reader.me() - km::first(command), km::count(command))
    );
    const km::Tensor<0> input(command.a0);
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!reader.reserve(TP_CB_IN, 1)) {
            return;
        }
        const uint32_t page = command.a6 + partition.start + tile;
        if (!km::read_pages(
            reader, command, input, page, 1, get_write_ptr(TP_CB_IN), TP_ASK_PORT
        )) {
            return;
        }
        noc_async_read_barrier();
        cb_push_back(TP_CB_IN, 1);
    }
}
