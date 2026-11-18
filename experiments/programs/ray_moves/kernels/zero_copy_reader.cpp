// The runtime's generic reader of a dynamic program (messages/reader.cpp) on the
// zero-copy copy of its messages, checking each message of the probe as it lands.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "zero_copy_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "bundle.h"

void kernel_main() {
    km::Reader reader;
    const uint32_t messages_handler = reader.arg(0);
    const uint32_t results_address = reader.arg(km::kMessagesArgs);
    const bool filled = km::messages_zero_fill(
        reader
    );  // before READY: nobody sends yet
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == messages_handler) {
            const uint32_t message_bytes = km::messages_args(reader).message_bytes;
            volatile tt_l1_ptr uint32_t* results = km::l1(results_address);
            uint32_t bad = 0, checked = 0;
            auto check = [&](uint32_t first, uint32_t count) {
                for (uint32_t index = 0; index < count; index++) {
                    const volatile uint32_t* message = (
                        km::l1(first + index * message_bytes)
                    );
                    bad += !bundle_valid(message, message_bytes / 4, command.a0);
                    checked++;
                }
                results[kResultReaderBad] = bad;
                results[kResultReaderChecked] = checked;
            };
            km::messages_reader(reader, command, filled, check);
        }
    });
}
