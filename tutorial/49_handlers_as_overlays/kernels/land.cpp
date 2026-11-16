// The land handler: a message is counted into this core's notes and consumed.
#include <stdint.h>

#include "komorebi/device/messages/km_messages.hpp"
#include "handlers.h"

// --8<-- [start:land]
void km_message(km::Messages& messages, km::Packet& packet) {
    uint32_t total = 0;
    for (uint32_t index = 0; index < packet.count; index++) {
        total += packet[index][kValue];
        messages.consume();
    }
    UNPACK(({
        volatile tt_l1_ptr uint32_t* notes = km::l1(messages.arg(0));
        notes[kNoteLanded] += packet.count;
        notes[kNoteTotal] += total;
    }));
}

// --8<-- [end:land]
