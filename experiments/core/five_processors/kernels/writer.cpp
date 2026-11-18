// The writer, on NCRISC and NoC 1: the movers' handlers as processor 1 (mover.hpp).
#include <stdint.h>

#include "mover.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        five::body(writer, kNcrisc, command);
        return true;
    });
}
