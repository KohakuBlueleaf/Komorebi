// Generic reader of a dynamic program: the dynamic handler takes this core's queue;
// every other command (the run's end command among them) is a NOP here.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    const uint32_t messages_handler = rd.arg(0);
    const bool filled = km::messages_zero_fill(rd);  // before READY: nobody sends yet
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) == messages_handler) {
            km::messages_reader(rd, r, filled);
        }
    });
}
