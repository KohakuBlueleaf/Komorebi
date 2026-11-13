// The reader kernel: it loads the overlay the command names. The reader is the one
// that loads every part of an overlay into its core, the writer's part too.
#include "komorebi/device/km_overlay.hpp"
#include "handlers.h"

void kernel_main() {
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        km::Overlay overlay(reader, command.a0);
        overlay.load();
    });
    // --8<-- [end:run]
}
