// The link experiment, writer: the link core holds the link's connection from start-up
// and runs its writer in a RUN, every other core the message writer with the outbox.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_link.hpp"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    const uint32_t messages_handlers = writer.arg(0);
    const bool linked = km::link_args(writer).core == writer.me();
    km::LinkWire wire(writer);
    if (linked) {
        wire.start(writer);
    }
    writer.run([&](const km_command_t& command) {
        if (!KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
            return true;
        }
        if (linked) {
            return km::link_sender(writer, command, wire, command.a5);
        }
        return km::messages_writer(writer, command, km::LinkOut(writer));
    });
    wire.close(writer);
}
