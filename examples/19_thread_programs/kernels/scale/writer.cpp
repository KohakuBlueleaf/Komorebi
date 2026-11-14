// The scale program's writer: SCALE writes the core's tiles to output tiles a3 + ...
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_program.hpp"
#include "../../programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != TP_HANDLER_SCALE) {
        return true;
    }
    const km::Partition partition = (
        km::partition(command.a1, writer.me() - km::first(command), km::count(command))
    );
    const km::Tensor<0> output(command.a2);
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!writer.wait_front(TP_CB_OUT, 1)) {
            return false;
        }
        const uint32_t page = command.a3 + partition.start + tile;
        if (!km::write_pages(
            writer, command, output, page, 1, get_read_ptr(TP_CB_OUT)
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(TP_CB_OUT, 1);
    }
    return true;
}
