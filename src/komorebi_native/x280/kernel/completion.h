/* Completion table: the busy mailbox slots, their sequence, request and deadline, and
 * the acknowledgement words the groups' leaders push into LIM, one per slot. */
#pragma once

#include "grid_port.h"
#include "kernel.h"
#include "ring.h"

/* The slot's ack line as one 64-bit word: the ack word, then the result word. */
static inline volatile const uint64_t* slot_ack(
    const kmd_t* dispatcher, uint32_t slot
) {
    return (
        (volatile const uint64_t*)(
            dispatcher->lim + KMD_LIM_ACKS + slot * KM_ACK_STRIDE
        )
    );
}

/* The lowest free slot, or KM_RING_SLOTS if every slot in use is busy. */
static inline uint32_t slot_free(const kmd_t* dispatcher) {
    for (uint32_t slot = 0; slot < dispatcher->depth; slot++) {
        if (!(dispatcher->busy & (1u << slot))) {
            return slot;
        }
    }
    return KM_RING_SLOTS;
}

#ifdef KMD_TRACE
/* Test builds: entry {kind | slot << 8 | source << 16, request, group, mtime} into the
 * trace part from its second line on; the first entry that does not fit, the header. */
static inline void trace(
    kmd_t* dispatcher, uint32_t kind, uint32_t slot, uint32_t request
) {
    if (
        64u + 16u * (dispatcher->traced + 1u) > KMD_TRACE_PART(dispatcher->upload_bytes)
    ) {
        if (dispatcher->untraced++ == 0u && dispatcher->upload_bytes >= 64u) {
            const uint32_t header[4] = {KMD_TRACE_MARK, dispatcher->traced, 1u, 0u};
            kmd_store_chunk((volatile uint8_t*)dispatcher->upload, header);
        }
        return;
    }
    const uint32_t entry[4] = {
        kind | slot << 8 | (uint32_t)dispatcher->slot_source[slot] << 16,
        request,
        dispatcher->slot_group[slot],
        (uint32_t)x280_mtime()
    };
    kmd_store_chunk(
        (volatile uint8_t*)dispatcher->upload + 64u + 16u * dispatcher->traced, entry
    );
    dispatcher->traced++;
}
#else
#define trace(dispatcher, kind, slot, request) ((void)0)
#endif

/* Marks `slot` busy with command `sequence` of `request` of source `source` for
 * `group`, submitted at `now` (trace entry `how`); `result`: it has KM_FLAG_RESULT. */
static inline void slot_take(
    kmd_t* dispatcher,
    uint32_t slot,
    uint32_t sequence,
    uint32_t request,
    uint32_t source,
    uint32_t group,
    uint64_t now,
    uint64_t ticks,
    uint32_t result,
    uint32_t how
) {
    dispatcher->slot_sequence[slot] = sequence;
    dispatcher->slot_request[slot] = request;
    dispatcher->slot_group[slot] = group;
    dispatcher->slot_source[slot] = (uint8_t)source;
    dispatcher->ring[source].busy_count++;
    dispatcher->slot_submitted[slot] = now;
    dispatcher->slot_deadline[slot] = now + ticks;
    dispatcher->busy |= 1u << slot;
    dispatcher->result_busy |= result ? 1u << slot : 0u;
    if (++dispatcher->busy_count > dispatcher->max_busy) {
        dispatcher->max_busy = dispatcher->busy_count;
    }
    trace(dispatcher, how, slot, request);
    (void)how;
}

/* Whether cores [first, first + count) are not all in the grid; written without a sum,
 * so no value of either wraps into the grid. */
static inline int group_outside(
    const kmd_t* dispatcher, uint32_t first, uint32_t count
) {
    return count == 0 || first > dispatcher->cores || count > dispatcher->cores - first;
}

/* The end (last core + 1) of a busy slot's group that shares a core with cores [first,
 * first + count), the furthest if several; 0 if none. Groups lie inside the grid. */
static inline uint32_t slots_overlap(
    const kmd_t* dispatcher, uint32_t first, uint32_t count
) {
    uint32_t end = 0;
    for (uint32_t slot = 0; slot < dispatcher->depth; slot++) {
        const uint32_t busy_group = dispatcher->slot_group[slot];
        const uint32_t busy_first = busy_group & 0xFFFFu;
        const uint32_t busy_end = busy_first + (busy_group >> 16);
        if (
            (dispatcher->busy & (1u << slot)) && busy_first < first + count &&
            first < busy_end && busy_end > end
        ) {
            end = busy_end;
        }
    }
    return end;
}

/* Group allocator: the lowest first such that cores [first, first + count) overlap no
 * busy slot's group, or KMD_NO_FIRST. Each round jumps past an overlapping group. */
static inline uint32_t group_alloc(const kmd_t* dispatcher, uint32_t count) {
    uint32_t first = 0;
    while (count <= dispatcher->cores - first) {
        const uint32_t end = slots_overlap(dispatcher, first, count);
        if (end == 0) {
            return first;
        }
        first = end;
    }
    return KMD_NO_FIRST;
}

/* The wake owed to each source rejected busy: a slot freed or the queue has room. */
static inline void sources_wake(kmd_t* dispatcher) {
    for (uint32_t source = 0; source < KMD_SOURCES; source++) {
        if (dispatcher->waits & (1u << source)) {
            dispatcher->ring[source].owed--;
            ring_push(&dispatcher->ring[source], 0, KMD_EVENT_WAKE, 0);
        }
    }
    dispatcher->waits = 0;
}

/* Sweeps busy slots' ack lines (8-byte loads); a slot showing its sequence is freed:
 * KM_EVENT_DONE {auxiliary first core | sequence low 16 << 8, ticks time/result}. */
KMD_HOT void slots_sweep(kmd_t* dispatcher) {
    x280_fence_ir();
    uint64_t now = 0;
    for (uint32_t slot = 0; slot < dispatcher->depth; slot++) {
        if (!(dispatcher->busy & (1u << slot))) {
            continue;
        }
        const uint64_t line = *slot_ack(dispatcher, slot);
        if ((uint32_t)line != KM_ACK(dispatcher->slot_sequence[slot])) {
            continue;
        }
        if (now == 0) {
            now = x280_mtime();
        }
        uint32_t ticks = (uint32_t)(now - dispatcher->slot_submitted[slot]);
        const uint32_t result = dispatcher->result_busy & (1u << slot);
        if (result) {
            dispatcher->result_busy &= ~(1u << slot);
            dispatcher->last_result = ticks = (uint32_t)(line >> 32);
            dispatcher->results++;
        }
        dispatcher->busy &= ~(1u << slot);
        dispatcher->busy_count--;
        trace(dispatcher, KMD_TRACE_FREE, slot, dispatcher->slot_request[slot]);
        kmd_ring_t* ring = &dispatcher->ring[dispatcher->slot_source[slot]];
        ring->busy_count--;
        if (dispatcher->messages.end ==
            slot + 1u) { /* a dynamic run's end command: no event */
            dispatcher->messages.end = 0;
        } else {
            const uint32_t auxiliary = (
                (dispatcher->slot_group[slot] & KM_EVENT_AUXILIARY_LOW) |
                (dispatcher->slot_sequence[slot] & KM_EVENT_SEQUENCE_MASK)
                    << KM_EVENT_SEQUENCE_SHIFT
            );
            ring_push(
                ring,
                dispatcher->slot_request[slot],
                KM_EVENT_DONE | auxiliary << 8,
                ticks
            );
        }
        dispatcher->last_event_mtime = now;
        if (dispatcher->waits) {
            sources_wake(dispatcher);
        }
    }
}

/* The busy slot whose deadline passed first, if any passed at `now`; else
 * KM_RING_SLOTS. A timeout names the command whose deadline expired first. */
static inline uint32_t slots_late(const kmd_t* dispatcher, uint64_t now) {
    uint32_t late = KM_RING_SLOTS;
    for (uint32_t slot = 0; slot < dispatcher->depth; slot++) {
        if (
            (dispatcher->busy & (1u << slot)) &&
            now > dispatcher->slot_deadline[slot] &&
            (
                late == KM_RING_SLOTS ||
                dispatcher->slot_deadline[slot] < dispatcher->slot_deadline[late]
            )
        ) {
            late = slot;
        }
    }
    return late;
}

/* Submits `line` into free `slot` for command `request` of `source` (trace entry
 * `how`): the next sequence (not 0, not the slot's last) and the landing words. */
KMD_HOT void command_submit(
    kmd_t* dispatcher,
    uint64_t* line,
    uint32_t slot,
    uint32_t request,
    uint32_t source,
    uint64_t now,
    uint64_t ticks,
    uint32_t how
) {
    const uint32_t header = (uint32_t)(line[0] >> 32);
    uint32_t sequence = ++dispatcher->sequence;
    while (sequence == 0 || sequence == dispatcher->slot_sequence[slot]) {
        sequence = ++dispatcher->sequence;
    }
    line[0] = sequence | ((uint64_t)((header & 0xFFFFu) | (slot << 16)) << 32);
    for (uint32_t landing = 2; landing < 8; landing += 2) {
        line[landing] = (line[landing] & ~(uint64_t)0xFFFFFFFFu) | sequence;
    }
    grid_store_command(dispatcher, slot, line);
    dispatcher->commands++;
    const uint32_t result = (header >> 8) & KM_FLAG_RESULT;
    slot_take(
        dispatcher,
        slot,
        sequence,
        request,
        source,
        (uint32_t)line[1],
        now,
        ticks,
        result,
        how
    );
}
