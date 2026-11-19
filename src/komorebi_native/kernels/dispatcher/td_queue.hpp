// The Tensix dispatcher's job queue, the X280's (x280/kernel/sources/queue.h): queued
// commands wait here until a slot is free and the allocator finds room.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "komorebi/abi/km_stream.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    // A waiting job in L1 (the block's kTbQueue, kTqJobBytes each): its line, then
    // these words.
    constexpr uint32_t kJobRequest = 16, kJobCount = 17, kJobPassed = 18;
    constexpr uint32_t kJobDeadlineLo = 19, kJobDeadlineHi = 20;

    struct Queue {
        uint32_t base = 0;  // the block's kTbQueue
        uint64_t used = 0;  // places holding a job
        uint8_t order[KM_QUEUE_JOBS];
        uint32_t length = 0;

        volatile tt_l1_ptr uint32_t* job(uint32_t place) const {
            return km::l1(base + place * kTqJobBytes);
        }

        // Puts command `line` (`request`, `count` cores, `deadline` ticks) last. The
        // caller checked that a place is free.
        void put(
            const uint32_t* line,
            uint32_t request,
            uint32_t count,
            uint64_t deadline
        ) {  // fmt: skip
            uint32_t place = 0;
            while (place < KM_QUEUE_JOBS - 1u && (used >> place & 1u)) {
                place++;
            }
            volatile tt_l1_ptr uint32_t* at = job(place);
            for (uint32_t word = 0; word < 16; word++) {
                at[word] = line[word];
            }
            at[kJobRequest] = request;
            at[kJobCount] = count;
            at[kJobPassed] = 0;
            at[kJobDeadlineLo] = static_cast<uint32_t>(deadline);
            at[kJobDeadlineHi] = static_cast<uint32_t>(deadline >> 32);
            used |= 1ull << place;
            order[length++] = static_cast<uint8_t>(place);
        }

        // Removes the job at `position` of the order; `before` loses its place.
        void remove(uint32_t position, uint64_t& before) {
            const uint64_t place = ~(1ull << order[position]);
            before &= place;
            used &= place;
            length--;
            for (uint32_t i = position; i < length; i++) {
                order[i] = order[i + 1u];
            }
        }
    };

}  // namespace td
