/* Completion table: the busy mailbox slots, their sequence, request and deadline, and
 * the acknowledgement words the teams' leaders push into LIM, one per slot. */
#pragma once

#include "grid_port.h"
#include "kernel.h"
#include "ring.h"

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

#ifdef KMD_TRACE
/* Test builds: entry {kind | slot << 8 | source << 16, req, team, mtime} into the trace
 * part from its second line on. The first entry that does not fit writes the header. */
static inline void trace(kmd_t* k, uint32_t kind, uint32_t slot, uint32_t req) {
    if (64u + 16u * (k->traced + 1u) > KMD_TRACE_PART(k->upload_bytes)) {
        if (k->untraced++ == 0u && k->upload_bytes >= 64u) {
            const uint32_t head[4] = {KMD_TRACE_MARK, k->traced, 1u, 0u};
            kmd_store_chunk((volatile uint8_t*)k->upload, head);
        }
        return;
    }
    const uint32_t e[4] = {
        kind | slot << 8 | (uint32_t)k->slot_src[slot] << 16,
        req,
        k->slot_team[slot],
        (uint32_t)x280_mtime()
    };
    kmd_store_chunk((volatile uint8_t*)k->upload + 64u + 16u * k->traced, e);
    k->traced++;
}
#else
#define trace(k, kind, slot, req) ((void)0)
#endif

/* Marks `slot` busy with record `seq` of command `req` of source `src` for `team`,
 * issued at `now` (trace entry `how`); `result`: the record has KM_F_RESULT. */
static inline void slot_take(
    kmd_t* k,
    uint32_t slot,
    uint32_t seq,
    uint32_t req,
    uint32_t src,
    uint32_t team,
    uint64_t now,
    uint64_t ticks,
    uint32_t result,
    uint32_t how
) {
    k->slot_seq[slot] = seq;
    k->slot_req[slot] = req;
    k->slot_team[slot] = team;
    k->slot_src[slot] = (uint8_t)src;
    k->ring[src].nbusy++;
    k->slot_t0[slot] = now;
    k->slot_deadline[slot] = now + ticks;
    k->busy |= 1u << slot;
    k->result_busy |= result ? 1u << slot : 0u;
    if (++k->nbusy > k->max_busy) {
        k->max_busy = k->nbusy;
    }
    trace(k, how, slot, req);
    (void)how;
}

/* Whether cores [first, first + count) are not all in the grid; written without a sum,
 * so no value of either wraps into the grid. */
static inline int team_outside(const kmd_t* k, uint32_t first, uint32_t count) {
    return count == 0 || first > k->cores || count > k->cores - first;
}

/* The end (last core + 1) of a busy slot's team that shares a core with cores [first,
 * first + count), the furthest if several; 0 if none. Teams lie inside the grid. */
static inline uint32_t slots_overlap(const kmd_t* k, uint32_t first, uint32_t count) {
    uint32_t end = 0;
    for (uint32_t j = 0; j < k->depth; j++) {
        const uint32_t t = k->slot_team[j], f = t & 0xFFFFu, e = f + (t >> 16);
        if ((k->busy & (1u << j)) && f < first + count && first < e && e > end) {
            end = e;
        }
    }
    return end;
}

/* Team allocator: the lowest first such that cores [first, first + count) overlap no
 * busy slot's team, or KMD_NO_FIRST. Each round jumps past an overlapping team. */
static inline uint32_t team_alloc(const kmd_t* k, uint32_t count) {
    uint32_t first = 0;
    while (count <= k->cores - first) {
        const uint32_t end = slots_overlap(k, first, count);
        if (end == 0) {
            return first;
        }
        first = end;
    }
    return KMD_NO_FIRST;
}

/* The wake owed to each source refused busy: a slot freed or the queue has room. */
static inline void sources_wake(kmd_t* k) {
    for (uint32_t src = 0; src < KMD_SOURCES; src++) {
        if (k->waits & (1u << src)) {
            k->ring[src].owed--;
            ring_push(&k->ring[src], 0, KMD_EV_WAKE, 0);
        }
    }
    k->waits = 0;
}

/* Sweeps busy slots' ack lines (one 8-byte load each); a slot showing its sequence is
 * freed with KM_EV_DONE {aux first core, ticks time or result} to its source. */
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
        const uint32_t result = k->result_busy & (1u << j);
        if (result) {
            k->result_busy &= ~(1u << j);
            k->last_result = ticks = (uint32_t)(line >> 32);
            k->results++;
        }
        k->busy &= ~(1u << j);
        k->nbusy--;
        trace(k, KMD_TRACE_FREE, j, k->slot_req[j]);
        kmd_ring_t* r = &k->ring[k->slot_src[j]];
        r->nbusy--;
        if (k->dyn.end == j + 1u) { /* a dynamic run's end record: no event */
            k->dyn.end = 0;
        } else {
            ring_push(
                r, k->slot_req[j], KM_EV_DONE | (k->slot_team[j] & 0xFFFFu) << 8, ticks
            );
        }
        k->t_last_event = now;
        if (k->waits) {
            sources_wake(k);
        }
    }
}

/* The busy slot whose deadline passed first, if any passed at `now`; else
 * KM_RING_SLOTS. A timeout names the record whose deadline expired first. */
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

/* Issues `l` into free `slot` for command `req` of source `src` (trace entry `how`):
 * the next sequence (not 0, not the slot's last) in seq and the landing words. */
KMD_HOT void record_issue(
    kmd_t* k,
    uint64_t* l,
    uint32_t slot,
    uint32_t req,
    uint32_t src,
    uint64_t now,
    uint64_t ticks,
    uint32_t how
) {
    const uint32_t hdr = (uint32_t)(l[0] >> 32);
    uint32_t seq = ++k->seq;
    while (seq == 0 || seq == k->slot_seq[slot]) {
        seq = ++k->seq;
    }
    l[0] = seq | ((uint64_t)((hdr & 0xFFFFu) | (slot << 16)) << 32);
    for (uint32_t j = 2; j < 8; j += 2) {
        l[j] = (l[j] & ~(uint64_t)0xFFFFFFFFu) | seq;
    }
    grid_store_record(k, slot, l);
    k->records++;
    const uint32_t result = (hdr >> 8) & KM_F_RESULT;
    slot_take(k, slot, seq, req, src, (uint32_t)l[1], now, ticks, result, how);
}
