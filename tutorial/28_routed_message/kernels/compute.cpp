// The compute kernel of a dynamic handler: rank 0 sends 12 messages, each to the core
// its value picks; every core adds up what it receives and keeps it in L1.
#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRoute) {
            return;
        }
        const uint32_t message_count = command.a0;
        const uint32_t notes_address = command.a1;
        km::Messages messages(compute, command);
        uint32_t message[kMessageWords] = {};
        uint32_t received = 0;
        uint32_t sum = 0;
        if (messages.seeding()) {
            if (messages.rank() == 0) {
                for (uint32_t value = 0; value < message_count; value++) {
                    message[kValue] = value;
                    messages.send_or_give_up(value * value % messages.count(), message);
                }
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                received++;
                sum += packet[index][kValue];
                messages.consume();
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* notes = km::l1(notes_address);
            notes[0] = received;
            notes[1] = sum;
        }));
    });
    // --8<-- [end:run]
}
