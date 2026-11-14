// Generic reader of a dynamic program: a dynamic handler takes this core's queue;
// every other command is a NOP here (KM_LINKS: link core 0 zeroes its link region).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#if defined(KM_LINKS)
#include "komorebi/device/km_link.hpp"
#endif
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    const uint32_t messages_handlers = reader.arg(0);
    const bool filled = km::messages_zero_fill(
        reader
    );  // before READY: nobody sends yet
#if defined(KM_LINKS)
    const km::LinkArgs link = km::link_args(reader);
    if (link.core == reader.args().me) {  // before its connection's open
        km::link_zero(link, km::messages_args(reader).message_bytes);
    }
#endif
    reader.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
            km::messages_reader(reader, command, filled);
        }
    });
}
