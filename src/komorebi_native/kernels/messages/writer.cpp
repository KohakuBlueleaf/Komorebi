// Generic writer of a dynamic program: a dynamic handler sends this core's messages
// (another island's: KM_LINKS by link core 0, KM_DELIVER by the link service).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#if defined(KM_LINKS)
#include "komorebi/device/links/km_link.hpp"
#elif defined(KM_DELIVER)
#include "komorebi/device/messages/km_deliver.hpp"
#endif
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    const uint32_t messages_handlers = writer.arg(0);
    writer.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
#if defined(KM_LINKS)
            return km::messages_writer(writer, command, km::LinkOut(writer));
#elif defined(KM_DELIVER)
            return km::messages_writer(writer, command, km::DeliverOut());
#else
            return km::messages_writer(writer, command);
#endif
        }
        return true;
    });
}
