// Message handlers example, compute: a thread looks its key up at its owner, goes on to
// the value's owner as another handler, and reports at its origin when done.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerLookup) {
            return;
        }
        const uint32_t threads = command.a0;
        const uint32_t steps = command.a1;
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(), count = messages.count();
        uint32_t lookups = 0, values = 0, reports = 0, own = 0, wrong = 0;
        uint32_t rejected = 0, probes = 0;
        uint64_t sum = 0;
        uint32_t message[kMessageWords] = {};
        if (messages.seeding()) {  // the seeds have the RUN command's handler: lookup
            const km::Partition partition = km::partition(threads, rank, count);
            for (uint32_t index = 0; index < partition.len; index++) {
                message[kMsgThread] = partition.start + index;
                message[kMsgOrigin] = rank;
                message[kMsgKey] = first_key(partition.start + index);
                message[kMsgSteps] = steps;
                rejected += !messages.send(message[kMsgKey] % count, message);
            }
            messages.seeded();
        }
        // One message goes on at `destination`, as `handler` there.
        auto go = [&](uint32_t destination, uint32_t handler) {
            own += destination == rank;
            messages.forward(destination, message, handler);
        };
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word <= kMsgProbes; word++) {
                    message[word] = incoming[word];
                }
                switch (messages.handler()) {
                    case kHandlerLookup:
                        lookups++;
                        message[kMsgProbes]++;
                        if (!table_holds(message[kMsgKey], rank, count)) {
                            own += (rank + 1) % count == rank;
                            messages.forward(
                                (rank + 1) % count, message
                            );  // still lookup
                            break;
                        }
                        message[kMsgValue] = table_value(message[kMsgKey]);
                        go(message[kMsgValue] % count, kHandlerValue);
                        break;
                    case kHandlerValue:
                        values++;
                        message[kMsgSum] += message[kMsgValue];
                        if (--message[kMsgSteps] != 0) {
                            message[kMsgKey] = next_key(message[kMsgValue]);
                            go(message[kMsgKey] % count, kHandlerLookup);
                        } else {
                            go(message[kMsgOrigin], kHandlerReport);
                        }
                        break;
                    case kHandlerReport:
                        reports++;
                        wrong += message[kMsgOrigin] != rank;
                        sum += message[kMsgSum];
                        probes += message[kMsgProbes];
                        messages.consume();
                        break;
                    default:
                        wrong++;
                        messages.consume();
                }
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* result_words = km::l1(results_address);
            result_words[kResLookups] = lookups;
            result_words[kResValues] = values;
            result_words[kResReports] = reports;
            result_words[kResOwn] = own;
            result_words[kResSumLo] = static_cast<uint32_t>(sum);
            result_words[kResSumHi] = static_cast<uint32_t>(sum >> 32);
            result_words[kResProbes] = probes;
            result_words[kResWrong] = wrong;
            result_words[kResRejected] = rejected;
            result_words[kResRun] = command.sequence;
        }));
    });
}
