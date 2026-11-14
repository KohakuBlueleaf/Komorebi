/* The job queue on core 0: commands with KM_FLAG_QUEUE from any source wait here until
 * a slot is free and the group allocator finds room (README.md, "Job queue"). */
#pragma once

#include "completion.h"
#include "kernel.h"

/* Publishes the queue's length and the engine's jobs taken in, for its ENQUEUE. */
static inline void queue_tell(kmd_t* dispatcher) {
    kmd_queue_t* queue = dispatcher->queue;
    dispatcher->queued = queue->length;
    *(volatile uint32_t*)(dispatcher->ering + KMD_ERING_QUEUED) = queue->length;
    *(volatile uint32_t*)(dispatcher->ering + KMD_ERING_TAKEN) = queue->taken;
}

/* Puts command `line` (command `request` of `source`, `count` cores, stream `tag`) at
 * the queue's end with its source's current deadline. The caller checked the room. */
static inline void queue_put(
    kmd_t* dispatcher,
    const uint64_t* line,
    uint32_t source,
    uint32_t request,
    uint32_t count,
    uint32_t tag
) {
    kmd_queue_t* queue = dispatcher->queue;
    uint32_t free_place = 0; /* the lowest free place: one exists, as n < KM_QUEUE_JOBS */
    while (free_place < KM_QUEUE_JOBS - 1u && (queue->used >> free_place & 1u)) {
        free_place++;
    }
    kmd_job_t* job = &queue->job[free_place];
    for (uint32_t word = 0; word < 8; word++) {
        job->line[word] = line[word];
    }
    job->ticks = dispatcher->ring[source].t_command;
    job->request = request;
    job->count = count;
    job->passed = 0;
    job->source = (uint8_t)source;
    job->tag = (uint16_t)tag;
    queue->used |= 1ull << free_place;
    queue->order[queue->length++] = (uint8_t)free_place;
    dispatcher->ring[source].queued_count++;
    queue->taken += source == KMD_SRC_ENGINE;
    queue_tell(dispatcher);
}

/* Removes the job at position `position` of the order; no line waits for it any more.
 */
static inline void queue_remove(kmd_t* dispatcher, uint32_t position) {
    kmd_queue_t* queue = dispatcher->queue;
    const uint64_t place = ~(1ull << queue->order[position]);
    for (uint32_t source = 0; source < KMD_SOURCES; source++) {
        dispatcher->ring[source].before &= place;
    }
    queue->used &= place;
    queue->length--;
    for (uint32_t i = position; i < queue->length; i++) {
        queue->order[i] = queue->order[i + 1u];
    }
}

/* The queue's turn (README.md): a count that found no room is not tried again in the
 * turn, as submitting only adds busy groups. Out of the loop's code: it is rare. */
static __attribute__((noinline)) void queue_turn(kmd_t* dispatcher, uint64_t now) {
    kmd_queue_t* queue = dispatcher->queue;
    const uint32_t look = queue->length < KM_QUEUE_LOOK ? queue->length : KM_QUEUE_LOOK;
    uint32_t position = 0, no_room = KMD_NO_FIRST, blocked = 0, submitted = 0;
    for (uint32_t examined = 0; examined < look && !blocked; examined++) {
        const uint32_t slot = slot_free(dispatcher);
        if (slot == KM_RING_SLOTS) {
            break; /* nothing is submitted and nobody is passed over in this turn */
        }
        kmd_job_t* job = &queue->job[queue->order[position]];
        const uint32_t first = (
            job->count < no_room ? group_alloc(dispatcher, job->count) : KMD_NO_FIRST
        );
        if (first == KMD_NO_FIRST) {
            no_room = job->count < no_room ? job->count : no_room;
            blocked = job->passed >= KM_QUEUE_PASSES;
            position++;
            continue;
        }
        for (uint32_t earlier = 0; earlier < position; earlier++) {
            blocked |= ++queue->job[queue->order[earlier]].passed >= KM_QUEUE_PASSES;
        }
        uint64_t* line = job->line;
        const uint32_t header = (uint32_t)(line[0] >> 32) & ~(KM_FLAG_QUEUE << 8);
        line[0] = (line[0] & 0xFFFFFFFFu) | (uint64_t)header << 32;
        line[1] = (
            (line[1] & ~(uint64_t)0xFFFFFFFFu) | (dispatcher->island_first + first) |
            job->count << 16
        );
        const uint32_t source = job->source, request = job->request;
        queue_remove(dispatcher, position);
        dispatcher->ring[source].queued_count--;
        command_submit(
            dispatcher, line, slot, request, source, now, job->ticks, KMD_TRACE_QUEUED
        );
        submitted++;
    }
    if (submitted) {
        queue_tell(dispatcher);
        if (dispatcher->waits) {
            sources_wake(dispatcher);
        }
    }
}

/* KMD_HANDLER_DROP: the engine's queued jobs of stream `tag` leave the queue, each
 * answered to the engine with KM_EVENT_REJECTED, reason KMD_REJECT_DROPPED. */
static inline void queue_drop(kmd_t* dispatcher, uint32_t tag) {
    kmd_queue_t* queue = dispatcher->queue;
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_ENGINE];
    for (uint32_t position = 0; position < queue->length;) {
        const kmd_job_t* job = &queue->job[queue->order[position]];
        if (job->source != KMD_SRC_ENGINE || job->tag != tag) {
            position++;
            continue;
        }
        ring_push(ring, job->request, KM_EVENT_REJECTED | (KMD_REJECT_DROPPED << 8), 0);
        ring->queued_count--;
        queue_remove(dispatcher, position);
    }
    queue_tell(dispatcher);
}
