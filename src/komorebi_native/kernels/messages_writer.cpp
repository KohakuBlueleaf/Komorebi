// Generic writer of a dynamic program: a dynamic handler sends this core's outgoing
// messages; every other command (the run's end command among them) completes as a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    const uint32_t messages_handlers = writer.arg(0);
    writer.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
            return km::messages_writer(writer, command);
        }
        return true;
    });
}
