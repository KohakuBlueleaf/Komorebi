// The link experiment, compute: origins on island 0 seed messages that go to a partner
// (on island 1 through the link core, or on their own island) and back, round trip
// after round trip, then end at their origin. The link core's own compute takes none.
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/km_link.hpp"
#include "komorebi/device/km_messages.hpp"
#include "komorebi/device/km_worker.hpp"
#include "cross.h"

void kernel_main() {
    km::Compute compute;
    const uint32_t results_address = compute.arg(km::kMessagesCArgs);
    const uint32_t island = compute.arg(km::kMessagesCArgs + 1);
    const uint32_t region = compute.arg(km::kMessagesCArgs + 2);  // the link's words
    compute.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerCross) {
            return;
        }
        const uint32_t origins = command.a0;
        const uint32_t per_origin = command.a1;
        const uint32_t trips = command.a2;
        const bool across = command.a3 == kModeAcross;
        const uint32_t link_rank = command.a4;
        km::Messages messages(compute, command);
        const uint32_t rank = messages.rank(), count = messages.count();
        const uint32_t ordinal = rank < link_rank ? rank : rank - 1;
        const bool origin = (
            island == 0 && rank != link_rank && rank < count && ordinal < origins
        );
        const uint32_t partner = across ? rank : (rank + count / 2) % count;
        uint32_t consumed = 0, visits = 0, rejected = 0, wrong = 0;
        uint32_t first_wall = 0, last_wall = 0;
        uint64_t ids = 0;
        uint32_t message[kMessageWords] = {};
        // Out to this origin's partner, or back to origin `from`: the destination.
        auto out = [&] {
            message[kMsgRoute] = (across ? 1u << km::kLinkIslandShift : 0u) | partner;
            return across ? link_rank : partner;
        };
        auto back = [&](uint32_t from) {
            message[kMsgRoute] = from;  // island 0
            return across ? link_rank : from;
        };
        if (messages.seeding() && origin) {
            first_wall = km::wall();
            for (uint32_t index = 0; index < per_origin; index++) {
                message[kMsgOrigin] = rank;
                message[kMsgTrips] = trips;
                message[kMsgId] = ordinal * per_origin + index;
                rejected += !messages.send(out(), message);
            }
        }
        messages.seeded();
        while (km::Packet packet = messages.next()) {
            for (uint32_t index = 0; index < packet.count; index++) {
                const volatile tt_l1_ptr uint32_t* incoming = packet[index];
                for (uint32_t word = 1; word < kMessageWords; word++) {
                    message[word] = incoming[word];
                }
                const uint32_t from = message[kMsgOrigin];
                const bool home = island == 0 && from == rank;
                const uint32_t route = island << km::kLinkIslandShift | rank;
                wrong += message[kMsgRoute] != route || rank == link_rank;
                if (!home) {
                    visits++;
                    messages.forward(back(from), message);
                    continue;
                }
                wrong += message[kMsgId] / per_origin != ordinal;
                if (--message[kMsgTrips] != 0) {
                    messages.forward(out(), message);
                    continue;
                }
                consumed++;
                ids += message[kMsgId];
                last_wall = km::wall();
                messages.consume();
            }
            messages.done(packet);
        }
        UNPACK(({
            volatile tt_l1_ptr uint32_t* result_words = km::l1(results_address);
            result_words[kResConsumed] = consumed;
            result_words[kResIdsLo] = static_cast<uint32_t>(ids);
            result_words[kResIdsHi] = static_cast<uint32_t>(ids >> 32);
            result_words[kResVisits] = visits;
            result_words[kResFirst] = first_wall;
            result_words[kResLast] = last_wall;
            result_words[kResRejected] = rejected;
            result_words[kResWrong] = wrong;
            result_words[kResRun] = command.sequence;
            for (uint32_t word = 0; rank == link_rank && word < kResLinkWords; word++) {
                result_words[kResWords + word] = km::l1(region)[word];
            }
        }));
    });
}
