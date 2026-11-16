// The direct ring, compute: tokens (with a body or not) hop over declared pairs, by a
// direct message (a forward when its slot is busy) or by the queues; then consumed.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "ring.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    const uint32_t seed_address = compute.arg(km::kMessagesCArgs + 1);  // a body
    const uint32_t body_bytes = (
        compute.arg(km::kMessagesCArgMessage) >> km::kMessagesCArgBodyShift
    );
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerRing) {
            return;
        }
        const uint32_t tokens = command.a0;
        const uint32_t hops = command.a1;
        const uint32_t mode = command.a2;
        const uint32_t pattern = command.a3;
        const uint32_t ranks = command.a4;
        const uint32_t records = command.a5;  // record room in the results area
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank();
        uint32_t visits = 0, consumed = 0, direct = 0, busy = 0;
        uint32_t token[kTokenWords] = {};
        volatile tt_l1_ptr uint32_t* results = km::l1(results_address);
        if (messages.seeding()) {
            volatile tt_l1_ptr uint32_t* seed = km::l1(seed_address);
            for (uint32_t index = 0; rank < ranks && index < tokens; index++) {
                const uint32_t id = rank * tokens + index;
                token[kTokId] = id;
                token[kTokHops] = hops;
                token[kTokState] = ring_seed(id);
                token[kTokSum] = 0;
                if (body_bytes == 0) {
                    messages.send(rank, token);  // its first visit is here
                    continue;
                }
                token[kTokChecksum] = 0;
                for (uint32_t word = 0; word < body_bytes / 4; word++) {
                    token[kTokChecksum] += ring_body(id, word);
                }
                PACK(({
                    for (uint32_t word = 0; word < body_bytes / 4; word++) {
                        seed[word] = ring_body(id, word);
                    }
                }));
                messages.send(rank, token, km::Body{seed_address, body_bytes});
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kTokenWords; word++) {
                    token[word] = incoming[word];
                }
                const km::Body body = packet.body(index);
                visits++;
                token[kTokSum] += rank;
                token[kTokState] = ring_next(token[kTokState]);
                if (token[kTokHops] == 0) {
                    if (consumed < records) {
                        PACK(({
                            volatile tt_l1_ptr uint32_t* record = (
                                results + kResRecords + consumed * kRecordWords
                            );
                            uint32_t checksum = 0;  // the body, checked where it ends
                            for (uint32_t word = 0; word < body.bytes / 4; word++) {
                                checksum += body.words()[word];
                            }
                            record[0] = token[kTokId] + 1;
                            record[1] = token[kTokState];
                            record[2] = token[kTokSum];
                            record[3] = (
                                body.bytes != body_bytes ||
                                checksum != token[kTokChecksum]
                            );
                        }));
                    }
                    consumed++;
                    messages.consume();
                    continue;
                }
                token[kTokHops]--;
                const uint32_t to = ring_to(rank, token[kTokState], pattern, ranks);
                if (body) {
                    if (mode == kModeDirect && messages.direct(to, token, body)) {
                        direct++;
                        continue;
                    }
                    busy += mode == kModeDirect;
                    messages.forward(to, token, body);
                    continue;
                }
                if (mode == kModeDirect && messages.direct(to, token)) {
                    direct++;
                    continue;
                }
                busy += mode == kModeDirect;
                messages.forward(to, token);
            }
            messages.done(packet);
        }
        PACK(({
            results[kResVisits] = visits;
            results[kResConsumed] = consumed;
            results[kResDirect] = direct;
            results[kResBusy] = busy;
            results[kResRun] = command.sequence;
        }));
    });
}
