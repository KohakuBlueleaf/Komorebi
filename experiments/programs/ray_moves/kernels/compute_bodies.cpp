// The transport probe's compute kernel on the runtime's bodies: a 64-byte line and a
// body forwarded as it arrived; the body's checksum is checked where it is consumed.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "bundle.h"

namespace {

    // Line words: word 0 the runtime's mark, word 1 its body bytes.
    constexpr uint32_t kLineId = 2;
    constexpr uint32_t kLineHops = 3;
    constexpr uint32_t kLineState = 4;
    constexpr uint32_t kLineChecksum = 5;
    constexpr uint32_t kLineWords = 16;

    // The sum of a body's payload words (bundle_payload of its id).
    uint32_t body_checksum(uint32_t id, uint32_t words) {
        uint32_t checksum = 0;
        for (uint32_t word = 0; word < words; word++) {
            checksum += bundle_payload(id, word);
        }
        return checksum;
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    const uint32_t seed_address = compute.arg(km::kMessagesCArgs + 1);
    const uint32_t body_bytes = (
        compute.arg(km::kMessagesCArgMessage) >> km::kMessagesCArgBodyShift
    );
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerBundle) {
            return;
        }
        const uint32_t message_count = command.a0;
        const uint32_t forwards = command.a1;
        const uint32_t words = body_bytes / 4;
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(), ranks = messages.count();
        uint32_t visits = 0, consumed = 0, bad = 0, rejected = 0, first = 0, last = 0;
        uint32_t check_cycles = 0, forward_cycles = 0, wait_cycles = 0;
        uint32_t line[kLineWords] = {};
        volatile tt_l1_ptr uint32_t* seed = km::l1(seed_address);
        if (messages.seeding()) {
            const km::Partition partition = km::partition(message_count, rank, ranks);
            for (uint32_t index = 0; index < partition.len; index++) {
                const uint32_t id = partition.start + index;
                PACK(({
                    for (uint32_t word = 0; word < words; word++) {
                        seed[word] = bundle_payload(id, word);
                    }
                }));
                line[kLineId] = id;
                line[kLineHops] = forwards;
                line[kLineState] = bundle_seed(id);
                line[kLineChecksum] = body_checksum(id, words);
                rejected += !messages.send(
                    bundle_destination(line[kLineState], ranks),
                    line,
                    km::Body{seed_address, body_bytes}
                );
            }
            messages.seeded();
        }
        uint32_t waited_from = km::wall();
        while (km::Packet packet = messages.next()) {
            wait_cycles += km::wall() - waited_from;
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kLineWords; word++) {
                    line[word] = incoming[word];
                }
                const km::Body body = packet.body(index);
                last = km::wall();
                first = visits == 0 ? last : first;
                visits++;
                if (line[kLineHops] == 0) {  // ends here: the body is checked once
                    PACK(({
                        uint32_t checksum = 0;
                        volatile tt_l1_ptr uint32_t* payload = body.words();
                        for (uint32_t word = 0; word < body.bytes / 4; word++) {
                            checksum += payload[word];
                        }
                        bad += (
                            checksum != line[kLineChecksum] ||
                            line[kLineId] >= message_count || body.bytes != body_bytes
                        );
                    }));
                    check_cycles += km::wall() - last;
                    consumed++;
                    messages.consume();
                    continue;
                }
                line[kLineHops]--;
                line[kLineState] = bundle_next(line[kLineState]);
                const uint32_t before = km::wall();
                messages.forward(
                    bundle_destination(line[kLineState], ranks), line, body
                );
                forward_cycles += km::wall() - before;
            }
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
    });
}
