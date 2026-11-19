// The Tensix dispatcher's busy slots and their sweep. A header of its own so that a
// later slice can run the sweep on another processor of the core.
#pragma once

#include <stdint.h>

#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    // The busy slots and their sweep (each ack word, then each deadline).
    struct Slots {
        uint32_t request[KM_RING_SLOTS];  // the host's command number; 0: free
        uint32_t sequence[KM_RING_SLOTS];
        uint32_t group[KM_RING_SLOTS];
        bool result[KM_RING_SLOTS];
        uint64_t submitted[KM_RING_SLOTS];
        uint32_t depth = 0, in_flight = 0, most = 0;
        volatile tt_l1_ptr uint32_t* ack = nullptr;

        void init(uint32_t slots, volatile tt_l1_ptr uint32_t* acks) {
            depth = slots;
            ack = acks;
            for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
                request[slot] = 0;
            }
        }

        // The first free slot, or depth when every slot is busy.
        uint32_t free_slot() const {
            uint32_t slot = 0;
            while (slot < depth && request[slot] != 0) {
                slot++;
            }
            return slot;
        }

        // Whether a busy slot's group shares a core with `other`.
        bool overlaps(uint32_t other) const {
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (request[slot] != 0 && km::groups_overlap(group[slot], other)) {
                    return true;
                }
            }
            return false;
        }

        void take(
            uint32_t slot,
            uint32_t req,
            uint32_t seq,
            uint32_t grp,
            bool res,
            uint64_t at
        ) {  // fmt: skip
            request[slot] = req;
            sequence[slot] = seq;
            group[slot] = grp;
            result[slot] = res;
            submitted[slot] = at;
            in_flight++;
            most = in_flight > most ? in_flight : most;
        }

        // Each acknowledged slot: done(request, status, ticks, result flag), then
        // freed. The first busy slot past its deadline: late(slot), and false.
        template <typename Done, typename Late>
        bool sweep(uint64_t now, uint64_t deadline, Done done, Late late) {
            for (uint32_t slot = 0; slot < depth; slot++) {
                if (request[slot] == 0) {
                    continue;
                }
                if (ack[slot * 4] == KM_ACK(sequence[slot])) {
                    const uint32_t ticks = (
                        result[slot]
                            ? ack[slot * 4 + 1]
                            : static_cast<uint32_t>(now - submitted[slot])
                    );
                    const uint32_t status = (
                        KM_EVENT_DONE | (group[slot] & 0xFFu) << 8 |
                        (sequence[slot] & KM_EVENT_SEQUENCE_MASK) << 16
                    );
                    const uint32_t req = request[slot];
                    request[slot] = 0;
                    in_flight--;
                    done(req, status, ticks, result[slot]);
                } else if (now - submitted[slot] > deadline) {
                    late(slot);
                    return false;
                }
            }
            return true;
        }

        // Busy slots 0 to 31 as bits.
        uint32_t busy_low() const {
            uint32_t bits = 0;
            for (uint32_t slot = 0; slot < depth && slot < 32; slot++) {
                bits |= (request[slot] != 0 ? 1u : 0u) << slot;
            }
            return bits;
        }
    };

}  // namespace td
