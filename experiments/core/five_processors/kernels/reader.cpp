// The reader, on BRISC and NoC 0: the movers' handlers as processor 0 (mover.hpp).
#include <stdint.h>

#include "mover.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        five::body(reader, kBrisc, command);
    });
}
