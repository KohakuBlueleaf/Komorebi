/* A source's two rings (kmd_ring_t): taking its commands and pushing its events. The
 * host and the engine on core 1 use the same code; only the memory differs. */
#pragma once

#include "kernel.h"

/* Copies command r->next into r->line if the source has stored it whole: 64-bit loads,
 * the rest of the line after a second load fence once seq shows it. Returns 1 or 0. */
static inline int ring_take(kmd_ring_t* r) {
    const uint32_t i = r->next;
    volatile const uint64_t* w = (
        (volatile const uint64_t*)(r->cmds +
        (i & (r->depth - 1u)) * sizeof(km_record_t))
    );
    uint64_t* l = r->line;
    x280_fence_ir();
    l[0] = w[0];
    if ((uint32_t)l[0] != i) {
        return 0;
    }
    x280_fence_ir();
    for (uint32_t j = 1; j < 8; j++) {
        l[j] = w[j];
    }
    /* KM_LANDED: land1..3 are the low halves of 64-bit words 2, 4 and 6. */
    return (uint32_t)l[2] == i && (uint32_t)l[4] == i && (uint32_t)l[6] == i;
}

/* The command was taken: count it and tell the source its line is free. */
static inline void ring_consumed(kmd_ring_t* r) {
    *r->consumed = r->next;
    r->next++;
}

/* Stores event ++pushed {req, status, ticks} as one 16-byte store (README.md: one
 * store per event); ring_flush makes it known. */
KMD_HOT void ring_push(kmd_ring_t* r, uint32_t req, uint32_t status, uint32_t ticks) {
    if (r->events == 0) {
        return;
    }
    const uint32_t index = ++r->pushed;
    uint32_t ev[4] __attribute__((aligned(16)));
    ev[0] = index;
    ev[1] = req;
    ev[2] = status;
    ev[3] = ticks;
    kmd_store_chunk(
        r->events + (index & (r->event_depth - 1u)) * sizeof(km_event_t), ev
    );
}

/* Whether the unread events, one per busy slot or queued job of this source, those it
 * is owed and one more fit its ring. The read count is loaded only when needed. */
static inline int ring_room(kmd_ring_t* r) {
    const uint32_t held = r->nbusy + r->nqueued + r->owed;
    if (r->pushed - r->read_n + held < r->event_depth) {
        return 1;
    }
    x280_fence_ir();
    r->read_n = *r->read;
    return r->pushed - r->read_n + held < r->event_depth;
}

/* Once per loop pass after events were pushed: one fence, then the event count. */
static inline void ring_flush(kmd_ring_t* r) {
    if (r->pushed != r->told_n && r->events != 0) {
        x280_fence_ow();
        *r->told = r->pushed;
        r->told_n = r->pushed;
    }
}
