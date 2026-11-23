// The frame program's reader: the passes' (handlers from 11), the fill's, SHOW and
// SHADE, LIGHT (show_, shade_, light_reader.hpp, in the fill's namespace), by handler.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_ask.hpp"

#include "grid.h"  // before the namespaces: one copy for all of them

#define RASTER_NO_ENTRY

namespace fill_k {
#include "reader.cpp"
#include "show_reader.hpp"
#include "shade_reader.hpp"
#include "light_reader.hpp"
}  // namespace fill_k

namespace pass_k {
#include "setup_reader.cpp"
}  // namespace pass_k

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == fill_k::kHandlerShow) {
            fill_k::show_reader_command(reader, command);
        } else if (km::handler(command) == fill_k::kHandlerShade) {
            fill_k::shade_reader_command(reader, command);
        } else if (km::handler(command) == fill_k::kHandlerLight) {
            fill_k::light_reader_command(reader, command);
        } else if (km::handler(command) >= pass_k::kHandlerLanes) {
            pass_k::setup_reader_command(reader, command);
        } else {
            fill_k::fill_reader_command(reader, command);
        }
    });
}
