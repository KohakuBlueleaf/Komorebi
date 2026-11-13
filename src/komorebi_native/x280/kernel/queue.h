/* The job queue on core 0: commands with KM_FLAG_QUEUE from any source wait here until
 * a slot is free and the group allocator finds room (README.md, "Job queue"). */
#pragma once

#include "completion.h"
#include "kernel.h"

/* Publishes the queue's length and the engine's jobs taken in, for its ENQUEUE. */
static inline void queue_tell(kmd_t* k) {
    kmd_queue_t* q = k->queue;
    k->queued = q->n;
    *(volatile uint32_t*)(k->ering + KMD_ERING_QUEUED) = q->n;
    *(volatile uint32_t*)(k->ering + KMD_ERING_TAKEN) = q->taken;
}

/* Puts command `l` (command `req` of `src`, `count` cores, stream `tag`) at the queue's
 * end with its source's current deadline. The caller checked the room. */
static inline void queue_put(
    kmd_t* k,
    const uint64_t* l,
    uint32_t src,
    uint32_t req,
    uint32_t count,
    uint32_t tag
) {
    kmd_queue_t* q = k->queue;
    uint32_t at = 0; /* the lowest free place: one exists, as n < KM_QUEUE_JOBS */
    while (at < KM_QUEUE_JOBS - 1u && (q->used >> at & 1u)) {
        at++;
    }
    kmd_job_t* j = &q->job[at];
    for (uint32_t i = 0; i < 8; i++) {
        j->line[i] = l[i];
    }
    j->ticks = k->ring[src].t_command;
    j->req = req;
    j->count = count;
    j->passed = 0;
    j->src = (uint8_t)src;
    j->tag = (uint16_t)tag;
    q->used |= 1ull << at;
    q->order[q->n++] = (uint8_t)at;
    k->ring[src].nqueued++;
    q->taken += src == KMD_SRC_ENGINE;
    queue_tell(k);
}

/* Removes the job at position `pos` of the order; no line waits for it any more. */
static inline void queue_remove(kmd_t* k, uint32_t pos) {
    kmd_queue_t* q = k->queue;
    const uint64_t place = ~(1ull << q->order[pos]);
    for (uint32_t src = 0; src < KMD_SOURCES; src++) {
        k->ring[src].before &= place;
    }
    q->used &= place;
    q->n--;
    for (uint32_t i = pos; i < q->n; i++) {
        q->order[i] = q->order[i + 1u];
    }
}

/* The queue's turn (README.md): a count that found no room is not tried again in the
 * turn, as submitting only adds busy groups. Out of the loop's code: it is rare. */
static __attribute__((noinline)) void queue_turn(kmd_t* k, uint64_t now) {
    kmd_queue_t* q = k->queue;
    const uint32_t look = q->n < KM_QUEUE_LOOK ? q->n : KM_QUEUE_LOOK;
    uint32_t pos = 0, no_room = KMD_NO_FIRST, blocked = 0, submitted = 0;
    for (uint32_t n = 0; n < look && !blocked; n++) {
        const uint32_t slot = slot_free(k);
        if (slot == KM_RING_SLOTS) {
            break; /* nothing is submitted and nobody is passed over in this turn */
        }
        kmd_job_t* j = &q->job[q->order[pos]];
        const uint32_t first = (
            j->count < no_room ? group_alloc(k, j->count) : KMD_NO_FIRST
        );
        if (first == KMD_NO_FIRST) {
            no_room = j->count < no_room ? j->count : no_room;
            blocked = j->passed >= KM_QUEUE_PASSES;
            pos++;
            continue;
        }
        for (uint32_t i = 0; i < pos; i++) {
            blocked |= ++q->job[q->order[i]].passed >= KM_QUEUE_PASSES;
        }
        uint64_t* l = j->line;
        const uint32_t hdr = (uint32_t)(l[0] >> 32) & ~(KM_FLAG_QUEUE << 8);
        l[0] = (l[0] & 0xFFFFFFFFu) | (uint64_t)hdr << 32;
        l[1] = (l[1] & ~(uint64_t)0xFFFFFFFFu) | first | j->count << 16;
        const uint32_t src = j->src, req = j->req;
        queue_remove(k, pos);
        k->ring[src].nqueued--;
        command_submit(k, l, slot, req, src, now, j->ticks, KMD_TRACE_QUEUED);
        submitted++;
    }
    if (submitted) {
        queue_tell(k);
        if (k->waits) {
            sources_wake(k);
        }
    }
}

/* KMD_HANDLER_DROP: the engine's queued jobs of stream `tag` leave the queue, each
 * answered to the engine with KM_EVENT_REJECTED, reason KMD_REJECT_DROPPED. */
static inline void queue_drop(kmd_t* k, uint32_t tag) {
    kmd_queue_t* q = k->queue;
    kmd_ring_t* r = &k->ring[KMD_SRC_ENGINE];
    for (uint32_t pos = 0; pos < q->n;) {
        const kmd_job_t* j = &q->job[q->order[pos]];
        if (j->src != KMD_SRC_ENGINE || j->tag != tag) {
            pos++;
            continue;
        }
        ring_push(r, j->req, KM_EVENT_REJECTED | (KMD_REJECT_DROPPED << 8), 0);
        r->nqueued--;
        queue_remove(k, pos);
    }
    queue_tell(k);
}
