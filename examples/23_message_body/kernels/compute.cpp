// Message body example, compute: a thread is one message with a body; at every hop the
// core changes each word of the body and forwards it, body and all, to the next rank.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "chain.h"

namespace {

    // This compute thread's share of `words` words (unpack, math, pack a third each).
    void third(uint32_t words, uint32_t& from, uint32_t& to) {
        uint32_t part = 0;
        MATH((part = 1));
        PACK((part = 2));
        from = words * part / 3;
        to = words * (part + 1) / 3;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const uint32_t area_address = compute.arg(km::kMessagesCArgs);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerChain) {
            return;
        }
        const uint32_t threads = command.a0;
        const uint32_t hops = command.a1;
        const uint32_t body_bytes = command.a2;
        const bool change = command.a3 != 0;
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(), count = messages.count();
        volatile tt_l1_ptr uint32_t* area = km::l1(area_address);
        uint32_t arrivals = 0, ended = 0, ended_hop = 0, ended_state = 0;
        uint32_t message[kMessageWords] = {};
        uint32_t from = 0, to = 0;
        if (messages.seeding()) {
            if (!messages.counter() && rank < threads) {  // thread `rank` starts here
                message[kMsgThread] = rank;
                message[kMsgHop] = 0;
                message[kMsgState] = rank + 1;
                third(body_bytes / 4, from, to);
                for (uint32_t word = from; word < to; word++) {
                    area[word] = chain_seed(rank, word);
                }
                if (body_bytes != 0) {
                    messages.send_or_give_up(
                        (rank + 1) % count, message, km::Body{area_address, body_bytes}
                    );
                } else {
                    messages.send_or_give_up((rank + 1) % count, message);
                }
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kMessageWords; word++) {
                    message[word] = incoming[word];
                }
                const km::Body body = packet.body(index);
                volatile tt_l1_ptr uint32_t* words = body.words();
                const uint32_t hop = message[kMsgHop];
                const uint32_t key = chain_key(hop, rank);
                arrivals++;
                message[kMsgState] = chain_state(message[kMsgState], key);
                message[kMsgHop] = hop + 1;
                third(body.bytes / 4, from, to);
                if (body && change) {
                    for (uint32_t word = from; word < to; word++) {
                        words[word] = chain_word(words[word], key, word);
                    }
                }
                if (hop + 1 == hops) {  // the thread ends here, its body in the area
                    for (uint32_t word = from; word < to; word++) {
                        area[word] = words[word];
                    }
                    ended = message[kMsgThread] + 1;
                    ended_hop = hop + 1;
                    ended_state = message[kMsgState];
                    messages.consume();
                    continue;
                }
                if (body) {
                    messages.forward((rank + 1) % count, message, body);
                } else {
                    messages.forward((rank + 1) % count, message);
                }
            }
            messages.done(packet);
        }
        UNPACK(({
            area[kResArrivals] = arrivals;
            area[kResThread] = ended;
            area[kResHop] = ended_hop;
            area[kResState] = ended_state;
            area[kResRun] = command.sequence;
        }));
    });
}
