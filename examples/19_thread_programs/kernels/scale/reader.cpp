// The scale program's reader: SCALE reads the core's partition of the input tiles.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../../programs.h"

void km_program(km::Reader& reader, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SCALE) {
        return;
    }
    constexpr auto input_args = TensorAccessorArgs<0>();
    const km::Partition partition = (
        km::partition(command.a1, reader.me() - km::first(command), km::count(command))
    );
    const auto input = TensorAccessor(input_args, command.a0);
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!reader.reserve(TP_CB_IN, 1)) {
            return;
        }
        noc_async_read_page(
            command.a6 + partition.start + tile, input, get_write_ptr(TP_CB_IN)
        );
        noc_async_read_barrier();
        cb_push_back(TP_CB_IN, 1);
    }
}
