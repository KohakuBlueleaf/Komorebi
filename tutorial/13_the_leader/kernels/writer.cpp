// The writer kernel: sets a result word, on every core or on one core of the group.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        const uint32_t rank = writer.me() - km::first(command);
        if (km::handler(command) == kHandlerEveryone) {
            writer.result(writer.me());
        }
        if (km::handler(command) == kHandlerOnlyOne && rank == command.a0) {
            writer.result(writer.me());
        }
        return true;
    });
    // --8<-- [end:run]
}
