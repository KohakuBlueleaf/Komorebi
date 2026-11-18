// The writer kernel: it calls the overlay the command names, and its answer is the
// command's result.
#include "komorebi/device/overlays/km_overlay.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        km::Overlay overlay(writer, command.a0);
        if (!overlay.load()) {  // waits until the reader has loaded it
            return false;
        }
        writer.result(overlay.call(Pair{command.a1, command.a2}));
        return true;
    });
    // --8<-- [end:run]
}
