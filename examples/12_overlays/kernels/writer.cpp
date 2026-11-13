// The writer: for scale, loads and calls the overlay's writer part, which writes the
// tiles compute made to the core's pages in DRAM.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_overlay.hpp"
#include "overlays.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRun || command.a0 != kOverlayScale) {
            return true;
        }
        km::Overlay scale(writer, kOverlayScale);
        if (!scale.load()) {  // meets the reader's load(), which reads this part
            return false;
        }
        return scale.call(DrainArgument{writer.me(), command.a3, command.a6}) ==
            command.a3;
    });
}
