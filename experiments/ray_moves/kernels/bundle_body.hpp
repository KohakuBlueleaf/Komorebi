// The transport probe's handler: a message hops between ranks from the destination
// table; the pack thread checks it (unless the reader does), forwards or consumes it.
#pragma once

#include <stdint.h>

#include "bundle.h"

template <typename ComputeType, typename Forward>
inline void bundle_run(
    ComputeType& compute,
    const km_command_t& command,
    uint32_t results_address,
    uint32_t seed_address,
    bool check,
    Forward forward
) {
    const uint32_t message_count = command.a0;
    const uint32_t forwards = command.a1;
    const uint32_t words = compute.arg(km::kMessagesCArgMessage) / 4;
    km::Messages messages(compute, command);
    const uint32_t rank = messages.rank(), ranks = messages.count();
    uint32_t visits = 0, consumed = 0, bad = 0, rejected = 0, first = 0, last = 0;
    uint32_t check_cycles = 0, forward_cycles = 0, wait_cycles = 0;
    if (messages.seeding()) {
        const km::Partition partition = km::partition(message_count, rank, ranks);
        volatile tt_l1_ptr uint32_t* seed = km::l1(seed_address);
        for (uint32_t index = 0; index < partition.len; index++) {
            const uint32_t id = partition.start + index;
            const uint32_t state = bundle_seed(id);
            PACK(({
                uint32_t checksum = 0;
                for (uint32_t word = kWordPayload; word < words; word++) {
                    seed[word] = bundle_payload(id, word);
                    checksum += seed[word];
                }
                seed[kWordId] = id;
                seed[kWordHops] = forwards;
                seed[kWordState] = state;
                seed[kWordChecksum] = checksum;
            }));
            const uint32_t destination = bundle_destination(state, ranks);
            rejected += !messages.send(destination, const_cast<const uint32_t*>(seed));
        }
        messages.seeded();
    }
    uint32_t waited_from = km::wall();
    while (km::Packet packet = messages.next()) {
        PACK(({
            wait_cycles += km::wall() - waited_from;
            for (uint32_t index = 0; index < packet.count; index++) {
                volatile tt_l1_ptr uint32_t*
                    incoming = const_cast<volatile tt_l1_ptr uint32_t*>(packet[index]);
                last = km::wall();
                first = visits == 0 ? last : first;
                visits++;
                if (check) {
                    bad += !bundle_valid(incoming, words, message_count);
                }
                check_cycles += km::wall() - last;
                const uint32_t left = incoming[kWordHops];
                if (left == 0) {
                    consumed++;
                    messages.consume();
                    continue;
                }
                const uint32_t state = bundle_next(incoming[kWordState]);
                incoming[kWordHops] = left - 1;
                incoming[kWordState] = state;
                const uint32_t before = km::wall();
                forward(messages, bundle_destination(state, ranks), incoming);
                forward_cycles += km::wall() - before;
            }
        }));
        messages.done(packet);
        waited_from = km::wall();
    }
    PACK(({
        volatile tt_l1_ptr uint32_t* result_words = km::l1(results_address);
        result_words[kResultVisits] = visits;
        result_words[kResultConsumed] = consumed;
        result_words[kResultBad] = bad;
        result_words[kResultRejected] = rejected;
        result_words[kResultFirst] = first;
        result_words[kResultLast] = last;
        result_words[kResultCheckCycles] = check_cycles;
        result_words[kResultForwardCycles] = forward_cycles;
        result_words[kResultWaitCycles] = wait_cycles;
        result_words[kResultRun] = command.sequence;
    }));
}
