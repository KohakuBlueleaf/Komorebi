// Generic writer of a dynamic program: a dynamic handler sends this core's outgoing
// messages (KM_LINKS: another island's through link core 0); others are NOPs.
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
    writer.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
#if defined(KM_LINKS)
            return km::messages_writer(writer, command, km::LinkOut(writer));
#else
            return km::messages_writer(writer, command);
#endif
        }
        return true;
    });
}
