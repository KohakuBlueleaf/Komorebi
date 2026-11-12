/* Completion table: the busy mailbox slots, their sequence, request and deadline, and
 * the acknowledgement words the teams' leaders push into LIM, one per slot. */
#pragma once

#include "event_ring.h"
#include "kernel.h"

static inline volatile const uint32_t* slot_ack(const kmd_t* k, uint32_t slot) {
    return (volatile const uint32_t*)(k->lim + KMD_LIM_ACKS + slot * KM_ACK_STRIDE);
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

/* Marks `slot` busy with record `seq` of request `req`, issued at `now`. */
static inline void slot_take(
    kmd_t* k, uint32_t slot, uint32_t seq, uint32_t req, uint64_t now, uint64_t ticks
) {
    k->slot_seq[slot] = seq;
    k->slot_req[slot] = req;
    k->slot_t0[slot] = now;
    k->slot_deadline[slot] = now + ticks;
    k->busy |= 1u << slot;
    uint32_t n = 0;
    for (uint32_t m = k->busy; m != 0; m &= m - 1u) {
        n++;
    }
    if (n > k->max_busy) {
        k->max_busy = n;
    }
}

/* Sweeps the ack words of the busy slots. A slot whose word shows its sequence is
 * freed and its request gets a KM_EV_DONE event. The kernel never writes ack words. */
static inline void slots_sweep(kmd_t* k) {
    x280_fence_ir();
    uint64_t now = 0;
    for (uint32_t j = 0; j < k->depth; j++) {
        if (!(k->busy & (1u << j)) || *slot_ack(k, j) != KM_ACK(k->slot_seq[j])) {
            continue;
        }
        if (now == 0) {
            now = x280_mtime();
        }
        k->busy &= ~(1u << j);
        events_push(k, k->slot_req[j], KM_EV_DONE, (uint32_t)(now - k->slot_t0[j]));
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
