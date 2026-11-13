// The compute kernel of a dynamic handler: a message visits a core, changes its value,
// and is forwarded to the core that the new value names, until its hops are used up.
#include "api/compute/common.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    // --8<-- [start:run]
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerWander) {
            return;
        }
        const uint32_t message_count = command.a0;
        const uint32_t hops = command.a1;
        const uint32_t notes_address = command.a2;
        km::Dyn dyn(compute, command);
        uint32_t message[kMessageWords] = {};
        uint32_t visits = 0;
        uint32_t ended_here = 0;
        if (dyn.seeding()) {
            if (dyn.rank() == 0) {
                for (uint32_t id = 0; id < message_count; id++) {
                    message[kHopsLeft] = hops;
                    message[kValue] = id + 1;
                    dyn.send(id % dyn.count(), message);
                }
            }
            dyn.seeded();
        }
        while (km::Packet packet = dyn.next()) {
            for (uint32_t index = 0; index < packet.n; index++) {
                message[kHopsLeft] = packet[index][kHopsLeft];
                message[kValue] = (packet[index][kValue] * 5 + 1) % 97;
                visits++;
                if (message[kHopsLeft] == 0) {
                    ended_here++;
                    dyn.consume();
                } else {
                    message[kHopsLeft] = message[kHopsLeft] - 1;
                    dyn.forward(message[kValue] % dyn.count(), message);
                }
            }
            dyn.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* notes = km::l1(notes_address);
            notes[0] = visits;
            notes[1] = ended_here;
        }));
    });
    // --8<-- [end:run]
}
