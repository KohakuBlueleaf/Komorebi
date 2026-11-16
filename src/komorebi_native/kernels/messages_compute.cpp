// The runtime's compute of a dynamic program whose handlers are overlay parts: the
// message loop; each group of a packet calls its handler's km_message (loaded if not).
#include <stdint.h>

#include "api/compute/common.h"
#include "komorebi/device/messages/km_messages.hpp"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"

#if !defined(KM_MESSAGE_HANDLERS)
#error "the host names the program's dynamic handlers (KM_MESSAGE_HANDLERS)"
#endif

namespace {

    // Overlay `number`'s part (the reader meets this load when the tile does not hold
    // it), called on `packet`. A load that gave up skips the call (recorded).
    void handle(
        km::Compute& compute,
        km::Messages& messages,
        km::Packet& packet,
        uint32_t number,
        uint32_t run
    ) {
        km::Overlay<km::Compute> part(compute, number);
        if (!part.load()) {
            UNPACK((km::messages_gave_up(
                compute.control(), km::kWhereMessagesLoad, number, run
            )));
            return;
        }
        part.call(km::MessageCall{&messages, &packet});
    }

}  // namespace

void kernel_main() {
    km::Compute compute;
    compute.run([&](const km_command_t& command) {
        const uint32_t seeds = (
            km::message_overlay(km::handler(command) << km::kMessagesHandlerShift)
        );
        if (seeds - 1u >= KM_DYNAMIC_MAX) {
            return;  // not a RUN command of the program's (its end among them)
        }
        km::Messages messages(compute, command);
        if (!messages.counter()) {  // the seeds: the RUN command's handler, no message
            km::Packet none{0, 0, 0};
            handle(compute, messages, none, seeds, command.sequence);
        }
        messages.seeded();
        while (km::Packet packet = messages.next()) {
            const uint32_t header = packet.base - km::kMessagesHeader;
            const volatile tt_l1_ptr uint8_t*
                bytes = (reinterpret_cast<const volatile tt_l1_ptr uint8_t*>(header));
            const uint32_t groups = km::l1(header)[km::kMessagesGroups];
            uint32_t first = 0;
            for (uint32_t group = 0; group < groups; group++) {
                const uint32_t end = bytes[km::kMessagesGroupEnds + group];
                if (
                    group >= km::kMessagesHandlersMax || end <= first ||
                    end > packet.count
                ) {
                    break;  // the reader's header: never so
                }
                km::Packet part = packet;
                part.count = end - first;
                part.order = bytes + km::kMessagesOrder + first;
                const uint32_t word = (
                    km::l1(packet.base + part.order[0] * packet.message_bytes)[0]
                );
                *packet.current = km::messages_handler(word);
                // a handler not in the list: one the run added (the reader checked)
                const uint32_t listed = km::message_overlay(word);
                handle(
                    compute,
                    messages,
                    part,
                    listed != 0 ? listed : km::messages_handler(word),
                    command.sequence
                );
                first = end;
            }
            messages.done(packet);
        }
    });
}
