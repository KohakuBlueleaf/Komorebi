// The sum program's writer: SUM writes the core's tile to output page a3 + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../two_programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != kHandlerSum) {
        return true;
    }
    constexpr auto output_args = TensorAccessorArgs<0>();
    if (!writer.wait_front(kCbOut, 1)) {
        return false;  // nothing to report: the tile is stopping
    }
    const auto output = TensorAccessor(output_args, command.a2);
    const uint32_t page = command.a3 + writer.me() - km::first(command);
    noc_async_write_page(page, output, get_read_ptr(kCbOut));
    noc_async_writes_flushed();
    cb_pop_front(kCbOut, 1);
    return true;
}
