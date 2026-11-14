// Generic writer of a dynamic program: a dynamic handler sends this core's outgoing
// messages (KM_LINKS: another island's through the link core); others are NOPs.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#if defined(KM_LINKS)
#include "komorebi/device/km_link.hpp"
#endif
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    const uint32_t messages_handlers = writer.arg(0);
#if defined(KM_LINKS)
    const bool linked = km::link_args(writer).core == writer.args().me;
    km::LinkWire wire(writer);
    if (linked) {
        wire.start(writer);
    }
#endif
    writer.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
#if defined(KM_LINKS)
            if (linked) {
                return km::link_sender(writer, command, wire, 0);
            }
            return km::messages_writer(writer, command, km::LinkOut(writer));
#else
            return km::messages_writer(writer, command);
#endif
        }
        return true;
    });
#if defined(KM_LINKS)
    wire.close(writer);
#endif
}
