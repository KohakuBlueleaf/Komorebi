// The lookup handler, an overlay part: at a key's owner, find the key's value and go on
// to the value's owner as value; a rank that does not hold the key asks the next one.
#include <stdint.h>

#include "komorebi/device/messages/km_messages.hpp"
#include "handlers.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    volatile tt_l1_ptr uint32_t* results = km::l1(messages.arg(0));
    const uint32_t rank = messages.rank(), count = messages.count();
    uint32_t message[kMessageWords] = {};
    if (messages.seeding()) {  // the RUN command's handler: a0 threads, a1 steps
        const km_command_t& command = messages.command();
        const km::Partition partition = km::partition(command.a0, rank, count);
        uint32_t rejected = 0;
        if (command.a2 != 0 && rank == 0) {  // a handler not the program's: it stops
            rejected += !messages.send(0, message, command.a2);
        }
        for (uint32_t index = 0; index < partition.len; index++) {
            message[kMsgThread] = partition.start + index;
            message[kMsgOrigin] = rank;
            message[kMsgKey] = first_key(partition.start + index);
            message[kMsgSteps] = command.a1;
            rejected += !messages.send(message[kMsgKey] % count, message);
        }
        UNPACK(({
            results[kResRejected] = rejected;
            results[kResRun] = command.sequence;
        }));
        return;
    }
    uint32_t own = 0;
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];
        for (uint32_t word = 1; word <= kMsgProbes; word++) {
            message[word] = incoming[word];
        }
        message[kMsgProbes]++;
        if (!table_holds(message[kMsgKey], rank, count)) {
            own += (rank + 1) % count == rank;
            messages.forward((rank + 1) % count, message);  // still lookup
            continue;
        }
        message[kMsgValue] = table_value(message[kMsgKey]);
        own += message[kMsgValue] % count == rank;
        messages.forward(message[kMsgValue] % count, message, kHandlerValue);
    }
    UNPACK(({
        results[kResLookups] += packet.count;
        results[kResOwn] += own;
    }));
}
