// The value handler, an overlay part: at the value's owner, add the value to the
// thread's sum; go on to the next key's owner as lookup, or home as report.
#include <stdint.h>

#include "komorebi/device/messages/km_messages.hpp"
#include "handlers.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    volatile tt_l1_ptr uint32_t* results = km::l1(messages.arg(0));
    const uint32_t rank = messages.rank(), count = messages.count();
    uint32_t message[kMessageWords] = {};
    uint32_t own = 0;
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];
        for (uint32_t word = 1; word <= kMsgProbes; word++) {
            message[word] = incoming[word];
        }
        message[kMsgSum] += message[kMsgValue];
        if (--message[kMsgSteps] != 0) {
            message[kMsgKey] = next_key(message[kMsgValue]);
            own += message[kMsgKey] % count == rank;
            messages.forward(message[kMsgKey] % count, message, kHandlerLookup);
        } else {
            own += message[kMsgOrigin] == rank;
            messages.forward(message[kMsgOrigin], message, kHandlerReport);
        }
    }
    UNPACK(({
        results[kResValues] += packet.count;
        results[kResOwn] += own;
    }));
}
