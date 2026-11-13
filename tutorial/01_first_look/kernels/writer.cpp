// The writer kernel: it adds the first two arguments of the command and answers.
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        const uint32_t left = command.a0;
        const uint32_t right = command.a1;
        writer.result(left + right);
        return true;
    });
}
