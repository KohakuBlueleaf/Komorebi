// The scale program's writer: SCALE writes the core's tiles to the output tiles of the
// same numbers.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../two_programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != kHandlerScale) {
        return true;
    }
    constexpr auto output_args = TensorAccessorArgs<0>();
    const km::Partition partition = (
        km::partition(command.a1, writer.me() - km::first(command), km::count(command))
    );
    const auto output = TensorAccessor(output_args, command.a2);
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!writer.wait_front(kCbOut, 1)) {
            return false;  // nothing to report: the tile is stopping
        }
        noc_async_write_page(
            command.a6 + partition.start + tile, output, get_read_ptr(kCbOut)
        );
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
    }
    return true;
}
