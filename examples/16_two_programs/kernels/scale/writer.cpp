// The scale program's writer: SCALE writes the core's tiles to output pages a3 + their
// places.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/core/km_program.hpp"
#include "../two_programs.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != kHandlerScale) {
        return true;
    }
    const km::Partition partition = (
        km::partition(command.a1, writer.me() - km::first(command), km::count(command))
    );
    const km::Tensor<0> output(command.a2);  // its reference word
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!writer.wait_front(kCbOut, 1)) {
            return false;  // nothing to report: the tile is stopping
        }
        if (!km::write_pages(
            writer,
            command,
            output,
            command.a3 + partition.start + tile,
            1,
            get_read_ptr(kCbOut)
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
    }
    return true;
}
