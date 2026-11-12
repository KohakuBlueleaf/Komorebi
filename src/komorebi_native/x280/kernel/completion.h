/* Completion table: the busy mailbox slots, their sequence, request and deadline, and
 * the acknowledgement words the teams' leaders push into LIM, one per slot. */
#pragma once

#include "event_ring.h"
#include "kernel.h"

/* The slot's ack line as one 64-bit word: the ack word, then the result word. */
static inline volatile const uint64_t* slot_ack(const kmd_t* k, uint32_t slot) {
    return (volatile const uint64_t*)(k->lim + KMD_LIM_ACKS + slot * KM_ACK_STRIDE);
}

/* The lowest free slot, or KM_RING_SLOTS if every slot in use is busy. */
static inline uint32_t slot_free(const kmd_t* k) {
    for (uint32_t j = 0; j < k->depth; j++) {
        if (!(k->busy & (1u << j))) {
            return j;
        }
    }
    return KM_RING_SLOTS;
}

/* Marks `slot` busy with record `seq` of request `req` for `team`, issued at `now`;
 * `result`: the record has KM_F_RESULT. */
static inline void slot_take(
    kmd_t* k,
    uint32_t slot,
    uint32_t seq,
    uint32_t req,
    uint32_t team,
    uint64_t now,
    uint64_t ticks,
    uint32_t result
) {
    k->slot_seq[slot] = seq;
    k->slot_req[slot] = req;
    k->slot_team[slot] = team;
    k->slot_t0[slot] = now;
    k->slot_deadline[slot] = now + ticks;
    k->busy |= 1u << slot;
    k->result_busy |= result ? 1u << slot : 0u;
    if (++k->nbusy > k->max_busy) {
        k->max_busy = k->nbusy;
    }
}

/* Whether a busy slot's team shares a core with `team` (first | count << 16). */
static inline int slots_overlap(const kmd_t* k, uint32_t team) {
    const uint32_t first = team & 0xFFFFu, end = first + (team >> 16);
    for (uint32_t j = 0; j < k->depth; j++) {
        const uint32_t t = k->slot_team[j], f = t & 0xFFFFu;
        if ((k->busy & (1u << j)) && f < end && first < f + (t >> 16)) {
            return 1;
        }
    }
    return 0;
}

/* Sweeps the busy slots' ack lines, one 8-byte load each; a slot showing its sequence
 * is freed with KM_EV_DONE (ticks: its time, or the result with KM_F_RESULT). */
static inline void slots_sweep(kmd_t* k) {
    x280_fence_ir();
    uint64_t now = 0;
    for (uint32_t j = 0; j < k->depth; j++) {
        if (!(k->busy & (1u << j))) {
            continue;
        }
        const uint64_t line = *slot_ack(k, j);
        if ((uint32_t)line != KM_ACK(k->slot_seq[j])) {
            continue;
        }
        if (now == 0) {
            now = x280_mtime();
        }
        uint32_t ticks = (uint32_t)(now - k->slot_t0[j]);
        if (k->result_busy & (1u << j)) {
            k->result_busy &= ~(1u << j);
            k->last_result = ticks = (uint32_t)(line >> 32);
            k->results++;
        }
        k->busy &= ~(1u << j);
        k->nbusy--;
        events_push(k, k->slot_req[j], KM_EV_DONE, ticks);
        k->t_last_event = now;
    }
}

/* The busy slot whose deadline passed first at `now`, or KM_RING_SLOTS. */
static inline uint32_t slots_late(const kmd_t* k, uint64_t now) {
    uint32_t late = KM_RING_SLOTS;
    for (uint32_t j = 0; j < k->depth; j++) {
        if ((k->busy & (1u << j)) && now > k->slot_deadline[j] &&
            (late == KM_RING_SLOTS || k->slot_deadline[j] < k->slot_deadline[late])) {
            late = j;
        }
    }
    return late;
}
