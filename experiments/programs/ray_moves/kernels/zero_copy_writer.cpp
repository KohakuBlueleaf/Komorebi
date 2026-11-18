// The runtime's generic writer of a dynamic program (messages/writer.cpp), built on
// the zero-copy copy of its messages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "zero_copy_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    const uint32_t messages_handler = writer.arg(0);
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == messages_handler) {
            return km::messages_writer(writer, command);
        }
        return true;
    });
}
