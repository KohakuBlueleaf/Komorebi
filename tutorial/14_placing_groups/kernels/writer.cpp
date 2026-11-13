// The writer kernel: every core of the group sets its own number as the result; the
// leader's reaches Python.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerWhere) {
            writer.result(writer.me());
        }
        return true;
    });
    // --8<-- [end:run]
}
