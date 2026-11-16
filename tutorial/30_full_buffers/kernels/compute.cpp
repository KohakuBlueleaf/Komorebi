// The compute kernel of a dynamic handler: rank 0 sends messages until send says no
// more, and every core consumes what it receives and notes how much quota it has left.
#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerFlood) {
            return;
        }
        const uint32_t notes_address = command.a0;
        km::Messages messages(compute, command);
        uint32_t message[kMessageWords] = {};
        uint32_t accepted = 0;
        uint32_t received = 0;
        if (messages.seeding()) {
            if (messages.rank() == 0) {
                const uint32_t attempts = 2 * messages.quota();
                for (uint32_t attempt = 0; attempt < attempts; attempt++) {
                    message[kValue] = attempt;
                    accepted += messages.send(attempt % messages.count(), message);
                }
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                received++;
                messages.consume();
            }
            messages.done(packet);
        }
        const uint32_t quota_left = messages.quota();
        UNPACK(({
            volatile tt_l1_ptr uint32_t* notes = km::l1(notes_address);
            notes[0] = accepted;
            notes[1] = received;
            notes[2] = quota_left;
        }));
    });
    // --8<-- [end:run]
}
