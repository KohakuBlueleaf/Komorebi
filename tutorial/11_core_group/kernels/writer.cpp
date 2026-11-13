// The writer kernel: every core of the group counts, a core of rank r up to r times the
// first argument, so that the cores take different times.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerWork) {
            const uint32_t rank = writer.me() - km::first(command);
            const uint32_t limit = rank * command.a0;
            volatile uint32_t number = 0;
            while (number < limit) {
                number = number + 1;
            }
        }
        return true;
    });
    // --8<-- [end:run]
}
