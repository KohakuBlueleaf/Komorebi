// The reader kernel. It has nothing to do.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {});
}
