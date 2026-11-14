// The link experiment, reader: the link core runs the link's reader in a RUN, every
// other core the runtime's message reader. The run's crossings come from its words.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_link.hpp"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "cross.h"

void kernel_main() {
    km::Reader reader;
    const uint32_t messages_handlers = reader.arg(0);
    const bool filled = km::messages_zero_fill(reader);  // before READY: nobody sends
    const bool linked = km::link_args(reader).core == reader.me();  // its writer zeroes
    reader.run([&](const km_command_t& command) {
        if (!KM_IS_DYNAMIC(messages_handlers, km::handler(command))) {
            return;
        }
        if (linked) {
            const uint32_t crossings = (
                command.a3 == kModeAcross ? command.a0 * command.a1 * command.a2 : 0
            );
            km::link_receiver(reader, command, km::LinkEnd{crossings, crossings});
        } else {
            km::messages_reader(reader, command, filled);
        }
    });
}
