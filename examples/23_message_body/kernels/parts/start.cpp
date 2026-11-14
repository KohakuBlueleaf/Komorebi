// The start handler as an overlay part (--send): the RUN's handler, so every core loads
// it; rank 0 seeds the thread and sends itself chain, loaded on first arrival.
#include <stdint.h>

#include "komorebi/device/km_messages.hpp"
#include "../chain.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    (void)packet;
    const km_command_t& command = messages.command();
    const uint32_t body_bytes = command.a2;
    const uint32_t rank = messages.rank();
    const uint32_t area_address = messages.arg(0);
    volatile tt_l1_ptr uint32_t* area = km::l1(area_address);
    UNPACK((area[kResRun] = command.sequence));
    if (!messages.seeding() || rank != 0) {
        return;
    }
    uint32_t part = 0;  // this compute thread's third of the body: unpack, math, pack
    MATH((part = 1));
    PACK((part = 2));
    uint32_t message[kMessageWords] = {};
    message[kMsgThread] = 0;
    message[kMsgState] = 1;
    const uint32_t words = body_bytes / 4;
    for (uint32_t word = words * part / 3; word < words * (part + 1) / 3; word++) {
        area[word] = chain_seed(0, word);
    }
    messages.send(0, message, kHandlerChain, km::Body{area_address, body_bytes});
}
