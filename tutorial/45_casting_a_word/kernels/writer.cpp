// The writer kernel. It has nothing to do: the reader writes each core's total.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        return true;
    });
}
