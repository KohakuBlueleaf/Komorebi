// The chain handler as an overlay part (km_message): the same hop as compute.cpp, its
// counts kept in the core's L1 area, since a part holds nothing between calls.
#include <stdint.h>

#include "komorebi/device/km_messages.hpp"
#include "../chain.h"

void km_message(km::Messages& messages, km::Packet& packet) {
    const km_command_t& command = messages.command();
    const uint32_t threads = command.a0, hops = command.a1, body_bytes = command.a2;
    const bool change = command.a3 != 0;
    const uint32_t rank = messages.rank(), count = messages.count();
    const uint32_t area_address = messages.arg(0);
    volatile tt_l1_ptr uint32_t* area = km::l1(area_address);
    uint32_t part = 0;  // this compute thread's third of a body: unpack, math, pack
    MATH((part = 1));
    PACK((part = 2));
    uint32_t message[kMessageWords] = {};
    if (messages.seeding()) {  // called once with no message: thread `rank` starts
        UNPACK((area[kResRun] = command.sequence));
        if (rank < threads) {
            message[kMsgThread] = rank;
            message[kMsgState] = rank + 1;
            const uint32_t words = body_bytes / 4;
            for (
                uint32_t word = words * part / 3; word < words * (part + 1) / 3; word++
            ) {
                area[word] = chain_seed(rank, word);
            }
            if (body_bytes != 0) {
                messages.send(
                    (rank + 1) % count, message, km::Body{area_address, body_bytes}
                );
            } else {
                messages.send((rank + 1) % count, message);
            }
        }
        return;
    }
    for (uint32_t index = 0; index < packet.count; index++) {
        const volatile tt_l1_ptr uint32_t* incoming = packet[index];
        for (uint32_t word = 1; word < kMessageWords; word++) {
            message[word] = incoming[word];
        }
        const km::Body body = packet.body(index);
        volatile tt_l1_ptr uint32_t* words = body.words();
        const uint32_t from = body.bytes / 4 * part / 3;
        const uint32_t to = body.bytes / 4 * (part + 1) / 3;
        const uint32_t hop = message[kMsgHop];
        const uint32_t key = chain_key(hop, rank);
        message[kMsgState] = chain_state(message[kMsgState], key);
        message[kMsgHop] = hop + 1;
        if (body && change) {
            for (uint32_t word = from; word < to; word++) {
                words[word] = chain_word(words[word], key, word);
            }
        }
        if (hop + 1 == hops) {  // the thread ends here, its body in the area
            for (uint32_t word = from; word < to; word++) {
                area[word] = words[word];
            }
            UNPACK(({
                area[kResThread] = message[kMsgThread] + 1;
                area[kResHop] = hop + 1;
                area[kResState] = message[kMsgState];
            }));
            messages.consume();
            continue;
        }
        if (body) {
            messages.forward((rank + 1) % count, message, body);
        } else {
            messages.forward((rank + 1) % count, message);
        }
    }
    UNPACK((area[kResArrivals] += packet.count));
}
