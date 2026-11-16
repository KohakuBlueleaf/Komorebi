// The sum program's writer: SUM writes the core's tile to output page a3 + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/core/km_program.hpp"
#include "../two_programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != kHandlerSum) {
        return true;
    }
    if (!writer.wait_front(kCbOut, 1)) {
        return false;  // nothing to report: the tile is stopping
    }
    const km::Tensor<0> output(command.a2);  // its reference word
    const uint32_t page = command.a3 + writer.me() - km::first(command);
    if (!km::write_pages(writer, command, output, page, 1, get_read_ptr(kCbOut))) {
        return false;
    }
    noc_async_writes_flushed();
    cb_pop_front(kCbOut, 1);
    return true;
}
