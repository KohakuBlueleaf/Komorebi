/* The thread's side of its command ring to the dispatch kernel: the host ring's
 * protocol (kernel README, "Protocol with the host"), with the placement as data. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"

/* The includer defines KMX_RING_FENCE_W and _R (fences), _SWEEP(p, n) (a hand-over
 * of a cached range), _NOW() (the timer) and _END() (KMX_OK or why waits end). */

#define KMX_RING_SPINS 256u /* polls between two checks of the clock and the stop */
#define KMX_STASH 32u       /* events kmx_wait set aside for kmx_poll */

typedef struct {
    volatile uint8_t* cmd;       /* cmd_depth slots of 64 bytes; 0: no ring */
    volatile uint32_t* consumed; /* the kernel's count of commands taken */
    volatile uint8_t* events;    /* event_depth events of 16 bytes */
    volatile uint64_t* read;     /* 64-byte line: word 0 the events the thread took */
    uint32_t cmd_depth, event_depth, cached;
    uint32_t next;    /* the next command's number, from 1 */
    uint32_t taken;   /* events taken from the ring */
    uint32_t stashed; /* events in stash[], oldest first */
    kmx_event stash[KMX_STASH];
} kmx_ring_t;

static inline int kmx_ring_pow2(uint32_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

/* Sets up `r` from the descriptor line: 1, or 0 when the line names no valid ring
 * (all-zero means none). Addresses are as this cluster sees them. */
static inline int kmx_ring_open(kmx_ring_t* r, const volatile kmx_ring_desc* d) {
    r->cmd = 0;
    r->next = 1;
    r->taken = 0;
    r->stashed = 0;
    if (d->cmd == 0 || !kmx_ring_pow2(d->cmd_depth) || !kmx_ring_pow2(d->event_depth) ||
        d->cmd_depth > KMX_RING_DEPTH_MAX || d->event_depth > KMX_RING_DEPTH_MAX ||
        (d->cmd & 63u) || (d->events & 15u) || (d->read & 63u) || (d->consumed & 3u) ||
        d->events == 0 || d->read == 0 || d->consumed == 0) {
        return 0;
    }
    r->consumed = (volatile uint32_t*)(uintptr_t)d->consumed;
    r->events = (volatile uint8_t*)(uintptr_t)d->events;
    r->read = (volatile uint64_t*)(uintptr_t)d->read;
    r->cmd_depth = d->cmd_depth;
    r->event_depth = d->event_depth;
    r->cached = (d->flags & KMX_RING_CACHED) != 0;
    r->cmd = (volatile uint8_t*)(uintptr_t)d->cmd;
    return 1;
}

/* Stores command r->next: the record's words first, then its number in seq and the
 * three landing words, so the kernel never sees the number over old words. */
static inline int kmx_ring_issue(
    kmx_ring_t* r, const km_record_t* rec, uint32_t* ticket
) {
    if (r->cmd == 0) {
        return KMX_STOPPED;
    }
    const uint32_t i = r->next;
    if (r->cached) {
        KMX_RING_SWEEP((uintptr_t)r->consumed, 4);
    }
    KMX_RING_FENCE_R();
    if ((int32_t)(i - r->cmd_depth - *r->consumed) > 0) {
        return KMX_FULL;
    }
    volatile uint32_t* w = (volatile uint32_t*)(r->cmd + (i % r->cmd_depth) * 64u);
    const uint32_t* src = (const uint32_t*)rec;
    for (uint32_t j = 0; j < 16; j++) {
        if (j % 4 != 0) {
            w[j] = src[j];
        }
    }
    KMX_RING_FENCE_W();
    for (uint32_t j = 0; j < 16; j += 4) {
        w[j] = i;
    }
    KMX_RING_FENCE_W();
    if (r->cached) {
        KMX_RING_SWEEP((uintptr_t)w, 64);
    }
    r->next = i + 1;
    *ticket = i;
    return KMX_OK;
}

/* Takes event r->taken + 1 if it is whole (its index read before and after its
 * fields), and tells the kernel through the read line. Returns 1 or 0. */
static inline int kmx_ring_take(kmx_ring_t* r, kmx_event* ev) {
    const uint32_t i = r->taken + 1;
    volatile const uint32_t* e = (
        (volatile const uint32_t*)(r->events + (i % r->event_depth) * 16u)
    );
    if (r->cached) {
        KMX_RING_SWEEP((uintptr_t)e, 16);
    }
    KMX_RING_FENCE_R();
    if (e[0] != i) {
        return 0;
    }
    KMX_RING_FENCE_R();
    const uint32_t req = e[1], status = e[2], ticks = e[3];
    KMX_RING_FENCE_R();
    if (e[0] != i) {
        return 0;
    }
    ev->ticket = req;
    ev->kind = status & 0xFFu;
    ev->aux = status >> 8;
    ev->ticks = ticks;
    r->taken = i;
    r->read[0] = i;
    for (uint32_t j = 1; j < 8; j++) {
        r->read[j] = 0;
    }
    KMX_RING_FENCE_W();
    if (r->cached) {
        KMX_RING_SWEEP((uintptr_t)r->read, 64);
    }
    return 1;
}

/* One event if any: a set-aside one first, else the ring's next. Never waits. */
static inline int kmx_ring_poll(kmx_ring_t* r, kmx_event* ev) {
    if (r->cmd == 0) {
        return KMX_STOPPED;
    }
    if (r->stashed != 0) {
        *ev = r->stash[0];
        r->stashed--;
        for (uint32_t j = 0; j < r->stashed; j++) {
            r->stash[j] = r->stash[j + 1];
        }
        return KMX_OK;
    }
    return kmx_ring_take(r, ev) ? KMX_OK : KMX_NONE;
}

/* The event of `ticket` if it came (KMX_OK), else KMX_NONE, or KMX_FULL as for
 * kmx_ring_wait; events of other tickets are set aside. Never waits. */
static inline int kmx_ring_look(kmx_ring_t* r, uint32_t ticket, kmx_event* ev) {
    if (r->cmd == 0) {
        return KMX_STOPPED;
    }
    for (uint32_t j = 0; j < r->stashed; j++) {
        if (r->stash[j].ticket == ticket) {
            *ev = r->stash[j];
            r->stashed--;
            for (uint32_t m = j; m < r->stashed; m++) {
                r->stash[m] = r->stash[m + 1];
            }
            return KMX_OK;
        }
    }
    kmx_event got;
    while (r->stashed < KMX_STASH && kmx_ring_take(r, &got)) {
        if (got.kind == KMX_EV_WAKE) {
            continue;
        }
        if (got.ticket == ticket) {
            *ev = got;
            return KMX_OK;
        }
        r->stash[r->stashed++] = got;
    }
    return r->stashed == KMX_STASH ? KMX_FULL : KMX_NONE;
}

/* The event of `ticket`, waiting until the timer passes `deadline`; events of other
 * tickets are set aside for kmx_poll (KMX_FULL while KMX_STASH are set aside). */
static inline int kmx_ring_wait(
    kmx_ring_t* r, uint32_t ticket, uint64_t deadline, kmx_event* ev
) {
    if (r->cmd == 0) {
        return KMX_STOPPED;
    }
    for (uint32_t j = 0; j < r->stashed; j++) {
        if (r->stash[j].ticket == ticket) {
            *ev = r->stash[j];
            r->stashed--;
            for (uint32_t m = j; m < r->stashed; m++) {
                r->stash[m] = r->stash[m + 1];
            }
            return KMX_OK;
        }
    }
    for (uint32_t spins = 1;; spins++) {
        kmx_event got;
        if (r->stashed == KMX_STASH) {
            return KMX_FULL; /* nothing is taken that could not be kept */
        }
        if (kmx_ring_take(r, &got)) {
            if (got.kind == KMX_EV_WAKE) {
                continue;
            }
            if (got.ticket == ticket) {
                *ev = got;
                return KMX_OK;
            }
            r->stash[r->stashed++] = got;
            continue;
        }
        if (spins % KMX_RING_SPINS == 0) {
            const int end = KMX_RING_END();
            if (end != KMX_OK) {
                return end;
            }
            if (KMX_RING_NOW() > deadline) {
                return KMX_LATE;
            }
        }
    }
}
