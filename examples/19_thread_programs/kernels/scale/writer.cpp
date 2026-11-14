// The scale program's writer: SCALE writes the core's tiles to output tiles a3 + ...
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../../programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SCALE) {
        return true;
    }
    constexpr auto output_args = TensorAccessorArgs<0>();
    const km::Partition partition = (
        km::partition(command.a1, writer.me() - km::first(command), km::count(command))
    );
    const auto output = TensorAccessor(output_args, command.a2);
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!writer.wait_front(TP_CB_OUT, 1)) {
            return false;
        }
        noc_async_write_page(
            command.a3 + partition.start + tile, output, get_read_ptr(TP_CB_OUT)
        );
        noc_async_writes_flushed();
        cb_pop_front(TP_CB_OUT, 1);
    }
    return true;
}
