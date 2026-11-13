// The writer kernel: it counts up to the first argument, which takes some time.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCount) {
            const uint32_t limit = command.a0;
            volatile uint32_t number = 0;
            while (number < limit) {
                number = number + 1;
            }
            writer.result(number);
        }
        return true;
    });
    // --8<-- [end:run]
}
