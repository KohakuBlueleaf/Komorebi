/* A source's two rings (kmd_ring_t): taking its commands and pushing its events. The
 * host and the engine on core 1 use the same code; only the memory differs. */
#pragma once

#include "../kernel.h"

/* Copies command ring->next into ring->line if the source stored it whole: 64-bit
 * loads, the rest after a second load fence once sequence shows it. 1 or 0. */
static inline int ring_take(kmd_ring_t* ring) {
    const uint32_t number = ring->next;
    volatile const uint64_t*
        words = (
            (volatile const uint64_t*)(
                ring->commands + (number & (ring->depth - 1u)) * sizeof(km_command_t)
            )
        );
    uint64_t* line = ring->line;
    x280_fence_ir();
    line[0] = words[0];
    if ((uint32_t)line[0] != number) {
        return 0;
    }
    x280_fence_ir();
    for (uint32_t word = 1; word < 8; word++) {
        line[word] = words[word];
    }
    /* KM_LANDED: land1..3 are the low halves of 64-bit words 2, 4 and 6. */
    return (
        (uint32_t)line[2] == number && (uint32_t)line[4] == number &&
        (uint32_t)line[6] == number
    );
}

/* The command was taken: count it and tell the source its line is free. */
static inline void ring_consumed(kmd_ring_t* ring) {
    *ring->consumed = ring->next;
    ring->next++;
}

/* Stores event ++pushed {request, status, ticks} as one 16-byte store (README.md: one
 * store per event); ring_flush makes it known. */
KMD_HOT void ring_push(
    kmd_ring_t* ring, uint32_t request, uint32_t status, uint32_t ticks
) {
    if (ring->events == 0) {
        return;
    }
    const uint32_t index = ++ring->pushed;
    uint32_t event[4] __attribute__((aligned(16)));
    event[0] = index;
    event[1] = request;
    event[2] = status;
    event[3] = ticks;
    kmd_store_chunk(
        ring->events + (index & (ring->event_depth - 1u)) * sizeof(km_event_t), event
    );
}

/* Whether the unread events, one per busy slot or queued job of this source, those it
 * is owed and one more fit its ring. The read count is loaded only when needed. */
static inline int ring_room(kmd_ring_t* ring) {
    const uint32_t held = ring->busy_count + ring->queued_count + ring->owed;
    if (ring->pushed - ring->read_count + held < ring->event_depth) {
        return 1;
    }
    x280_fence_ir();
    ring->read_count = *ring->read;
    return ring->pushed - ring->read_count + held < ring->event_depth;
}

/* Once per loop pass after events were pushed: one fence, then the event count. */
static inline void ring_flush(kmd_ring_t* ring) {
    if (ring->pushed != ring->told_count && ring->events != 0) {
        x280_fence_ow();
        *ring->told = ring->pushed;
        ring->told_count = ring->pushed;
    }
}
