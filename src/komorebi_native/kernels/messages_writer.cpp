// Generic writer of a dynamic program: the dynamic handler sends this core's outgoing
// messages; every other command (the run's end command among them) completes as a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer wr;
    const uint32_t messages_handler = wr.arg(0);
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) == messages_handler) {
            return km::messages_writer(wr, r);
        }
        return true;
    });
}
