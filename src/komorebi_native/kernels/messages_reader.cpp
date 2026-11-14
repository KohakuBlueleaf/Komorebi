// Generic reader of a dynamic program: a dynamic handler takes this core's queue (the
// link core's reader with KM_LINKS); every other command is a NOP here.
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
    const bool linked = km::link_args(reader).core == reader.args().me;
#endif
    reader.run([&](const km_command_t& command) {
        if (KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
#if defined(KM_LINKS)
            if (linked) {
                km::link_receiver(reader, command, km::LinkEnd{0, 0});
                return;
            }
#endif
            km::messages_reader(reader, command, filled);
        }
    });
}
