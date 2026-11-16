// The climb handler: a message adds the rank to its value and goes on to the next rank;
// when its hops are used up it goes to this core again, as land. The RUN command seeds.
#include <stdint.h>

#include "komorebi/device/messages/km_messages.hpp"
#include "handlers.h"

// --8<-- [start:climb]
void km_message(km::Messages& messages, km::Packet& packet) {
    uint32_t message[kMessageWords] = {};
    if (messages.seeding()) {
        const km_command_t& command = messages.command();
        if (messages.rank() == 0) {
            for (uint32_t id = 0; id < command.a0; id++) {
                message[kHopsLeft] = command.a1;
                message[kValue] = id + 1;
                messages.send(id % messages.count(), message);  // as climb
            }
        }
        return;
    }
    for (uint32_t index = 0; index < packet.count; index++) {
        message[kHopsLeft] = packet[index][kHopsLeft];
        message[kValue] = packet[index][kValue] + messages.rank();
        if (message[kHopsLeft] == 0) {
            messages.forward(messages.rank(), message, kHandlerLand);
        } else {
            message[kHopsLeft] = message[kHopsLeft] - 1;
            messages.forward((messages.rank() + 1) % messages.count(), message);
        }
    }
    UNPACK((km::l1(messages.arg(0))[kNoteClimbs] += packet.count));
}

// --8<-- [end:climb]
