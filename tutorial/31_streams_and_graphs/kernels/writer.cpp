// The writer kernel: the handler of the command decides what it computes.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        const uint32_t left = command.a0;
        const uint32_t right = command.a1;
        if (km::handler(command) == kHandlerAdd) {
            writer.result(left + right);
        }
        if (km::handler(command) == kHandlerMultiply) {
            writer.result(left * right);
        }
        return true;
    });
    // --8<-- [end:run]
}
