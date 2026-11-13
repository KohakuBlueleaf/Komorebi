// The writer kernel: ADD answers with the sum of its two arguments.
#include "handlers.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerAdd) {
            writer.result(command.a0 + command.a1);
        }
        return true;
    });
}
