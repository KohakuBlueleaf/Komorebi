// The runtime's generic writer of a dynamic program (messages_writer.cpp), built on
// the copy of its messages for page-sized messages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "page_messages.hpp"
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
