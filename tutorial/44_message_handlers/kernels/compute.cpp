// The compute kernel of two dynamic handlers: a message climbs from rank to rank,
// adding each rank to its value, then lands on the core where its hops ran out.
#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerClimb) {
            return;
        }
        const uint32_t message_count = command.a0;
        const uint32_t hops = command.a1;
        const uint32_t notes_address = command.a2;
        km::Messages messages(compute, command);
        uint32_t message[kMessageWords] = {};
        uint32_t climbs = 0, landed = 0, total = 0;
        if (messages.seeding()) {
            if (messages.rank() == 0) {
                for (uint32_t id = 0; id < message_count; id++) {
                    message[kHopsLeft] = hops;
                    message[kValue] = id + 1;
                    messages.send(id % messages.count(), message);  // as climb
                }
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                message[kHopsLeft] = packet[index][kHopsLeft];
                message[kValue] = packet[index][kValue];
                if (messages.handler() == kHandlerLand) {
                    landed++;
                    total += message[kValue];
                    messages.consume();
                    continue;
                }
                climbs++;
                message[kValue] += messages.rank();
                if (message[kHopsLeft] == 0) {
                    messages.forward(messages.rank(), message, kHandlerLand);
                } else {
                    message[kHopsLeft] = message[kHopsLeft] - 1;
                    messages.forward((messages.rank() + 1) % messages.count(), message);
                }
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* notes = km::l1(notes_address);
            notes[0] = climbs;
            notes[1] = landed;
            notes[2] = total;
        }));
    });
    // --8<-- [end:run]
}
