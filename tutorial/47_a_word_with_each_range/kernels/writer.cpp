// The writer kernel. It has nothing to do yet, but it must say that it is done.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        return true;
    });
    // --8<-- [end:run]
}
