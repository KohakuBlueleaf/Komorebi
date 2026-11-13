// The writer kernel: it answers every command with the number 42.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        writer.result(42);
        return true;
    });
    // --8<-- [end:run]
}
