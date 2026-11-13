// The frame program's reader: the passes' (handlers from 11) and the fill's, each
// file included in a namespace of its own without its entry, chosen by handler.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

#define RASTER_NO_ENTRY

namespace fill_k {
#include "reader.cpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_reader.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) >= pass_k::kHandlerLanes) {
            pass_k::setup_reader_command(reader, command);
        } else {
            fill_k::fill_reader_command(reader, command);
        }
    });
}
