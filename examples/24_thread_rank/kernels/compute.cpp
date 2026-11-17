// Example 24, compute: a core seeds messages to the X280 ranks; each that comes back is
// stepped and goes on, or is consumed when its hops are used up.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "rank.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerGrid) {
            return;
        }
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(), count = messages.count();
        const uint32_t x280 = messages.x280();
        uint32_t consumed = 0, visits = 0, to_x280 = 0, rejected = 0, mixed = 0;
        uint64_t sum = 0;
        uint32_t message[kMessageWords] = {};
        if (messages.seeding()) {
            for (uint32_t index = 0; index < command.a0 && x280 != 0; index++) {
                const uint32_t id = rank * command.a0 + index;
                message[kMsgId] = id;
                const uint32_t odd = (command.a2 & kSeedEven) != 0 ? 0 : id & 1;
                message[kMsgHops] = command.a1 + odd;
                message[kMsgValue] = seed_value(id);
                const bool to_cores = (command.a2 & kSeedToCores) != 0;
                const bool sent = (
                    to_cores
                        ? messages.send(
                            message[kMsgValue] % count, message, kHandlerGrid
                        )
                        : messages.send(
                            count + message[kMsgValue] % x280, message, kHandlerX280
                        )
                );
                rejected += sent ? 0 : 1;
                to_x280 += sent && !to_cores ? 1 : 0;
            }
            messages.seeded();
        }
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kMessageWords; word++) {
                    message[word] = incoming[word];
                }
                visits++;
                message[kMsgValue] = grid_step(message[kMsgValue]);
                if (message[kMsgHops] == 0) {
                    consumed++;
                    sum += message[kMsgValue];
                    mixed ^= end_mix(message[kMsgId], message[kMsgValue]);
                    messages.consume();
                    continue;
                }
                message[kMsgHops]--;
                messages.forward(
                    count + message[kMsgValue] % x280, message, kHandlerX280
                );
                to_x280++;
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* result_words = km::l1(results_address);
            result_words[kResConsumed] = consumed;
            result_words[kResSumLo] = static_cast<uint32_t>(sum);
            result_words[kResSumHi] = static_cast<uint32_t>(sum >> 32);
            result_words[kResXor] = mixed;
            result_words[kResVisits] = visits;
            result_words[kResToX280] = to_x280;
            result_words[kResRejected] = rejected;
            result_words[kResRun] = command.sequence;
        }));
    });
}
