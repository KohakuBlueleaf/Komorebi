// Forward example, compute: each visit advances the message's generator and adds the
// state to its sum; forwarded until none are left, then consumed into core totals.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "forward.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerForward) {
            return;
        }
        const uint32_t message_count = command.a0;
        const uint32_t forwards = command.a1;
        const uint32_t mode = command.a2;
        const uint32_t target_rank = command.a3;
        const uint32_t stall_rank_plus_one = command.a4;
        const uint32_t probe_rank_plus_one = command.a5;
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(),
                       count = messages.count();  // counter tiles: none
        uint32_t consumed = 0, visits = 0, nulls = 0, sent = 0, rejected = 0;
        uint32_t last_consume = 0;
        uint64_t total = 0;
        uint32_t message[kMessageWords] = {};
        if (messages.seeding()) {
            const km::Partition partition = km::partition(message_count, rank, count);
            for (uint32_t index = 0; index < partition.len; index++) {
                message[kMsgId] = partition.start + index;
                message[kMsgForwards] = forwards;
                message[kMsgState] = forward_seed(partition.start + index);
                message[kMsgSum] = 0;
                rejected += !messages.send(rank, message);
            }
            if (probe_rank_plus_one == rank + 1) {  // send until the quota is gone
                message[kMsgId] = kNullId;
                message[kMsgForwards] = 0;
                while (messages.send(rank, message)) {
                    sent++;
                }
                rejected++;
            }
            messages.seeded();
        }
        bool stall = stall_rank_plus_one == rank + 1;
        while (km::Packet packet = messages.next()) {
            if (stall) {  // test: this core never finishes its first packet
                messages.wait_stop();
                stall = false;
            }
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kMessageWords; word++) {
                    message[word] = incoming[word];
                }
                visits++;
                if (message[kMsgId] == kNullId) {
                    nulls++;
                    messages.consume();
                    continue;
                }
                message[kMsgState] = forward_next(message[kMsgState]);
                message[kMsgSum] += message[kMsgState];
                if (message[kMsgForwards] == 0) {
                    consumed++;
                    total += message[kMsgSum];
                    last_consume = km::wall();
                    messages.consume();
                    continue;
                }
                message[kMsgForwards]--;
                messages.forward(
                    mode == kModeOneCore ? target_rank : message[kMsgState] % count,
                    message
                );
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* result_words = km::l1(results_address);
            result_words[kResConsumed] = consumed;
            result_words[kResTotalLo] = static_cast<uint32_t>(total);
            result_words[kResTotalHi] = static_cast<uint32_t>(total >> 32);
            result_words[kResVisits] = visits;
            result_words[kResLastConsume] = last_consume;
            result_words[kResNull] = nulls;
            result_words[kResSent] = sent;
            result_words[kResRejected] = rejected;
            result_words[kResQuota] = messages.quota();
            result_words[kResRun] = command.sequence;
        }));
    });
}
