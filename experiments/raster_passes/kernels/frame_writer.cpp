// The frame program's writer: the passes' (handlers from 11) and the fill's, each
// file included in a namespace of its own without its entry, chosen by handler.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#define RASTER_NO_ENTRY

namespace fill_k {
#include "writer.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_writer.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) >= pass_k::kHandlerLanes) {
            return pass_k::setup_writer_command(writer, command);
        }
        return fill_k::fill_writer_command(writer, command);
    });
}
