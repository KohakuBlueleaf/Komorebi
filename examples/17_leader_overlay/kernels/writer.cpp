// The writer: no part in this program; the reader writes the results.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        return true;
    });
}
