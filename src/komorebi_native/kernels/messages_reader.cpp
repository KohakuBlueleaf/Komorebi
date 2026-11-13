// Generic reader of a dynamic program: the dynamic handler takes this core's queue;
// every other command (the run's end command among them) is a NOP here.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    const uint32_t messages_handler = reader.arg(0);
    const bool filled = km::messages_zero_fill(
        reader
    );  // before READY: nobody sends yet
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == messages_handler) {
            km::messages_reader(reader, command, filled);
        }
    });
}
