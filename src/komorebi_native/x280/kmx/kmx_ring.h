/* The thread's side of its command ring to the dispatcher: the host ring's
 * protocol (kernel README, "Protocol with the host"), with the placement as data. */
#pragma once

#include <stdint.h>

#include "kmx_layout.h"
#include "komorebi/abi/km_abi.h"

/* The includer defines KMX_RING_FENCE_W and _R (fences), _SWEEP(address, bytes) (flush
 * or invalidate a cached range), _NOW() (the timer), _END() (KMX_OK or why it ends) */

#define KMX_RING_SPINS 256u /* polls between two checks of the clock and the stop */
#define KMX_STASH 32u       /* events kmx_wait set aside for kmx_poll */

typedef struct {
    volatile uint8_t* commands;  /* command_depth slots of 64 bytes; 0: no ring */
    volatile uint32_t* consumed; /* the kernel's count of commands taken */
    volatile uint8_t* events;    /* event_depth events of 16 bytes */
    volatile uint64_t* read;     /* 64-byte line: word 0 the events the thread took */
    uint32_t command_depth, event_depth, cached;
    uint32_t next;    /* the next command's number, from 1 */
    uint32_t taken;   /* events taken from the ring */
    uint32_t stashed; /* events in stash[], oldest first */
    kmx_event stash[KMX_STASH];
} kmx_ring_t;

static inline int kmx_ring_pow2(uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

/* Sets up `ring` from the descriptor line: 1, or 0 when the line names no valid ring
 * (all-zero means none). Addresses are as this cluster sees them. */
static inline int kmx_ring_open(
    kmx_ring_t* ring, const volatile kmx_ring_descriptor* descriptor
) {
    ring->commands = 0;
    ring->next = 1;
    ring->taken = 0;
    ring->stashed = 0;
    if (
        descriptor->cmd == 0 || !kmx_ring_pow2(descriptor->command_depth) ||
        !kmx_ring_pow2(descriptor->event_depth) ||
        descriptor->command_depth > KMX_RING_DEPTH_MAX ||
        descriptor->event_depth > KMX_RING_DEPTH_MAX || (descriptor->cmd & 63u) ||
        (descriptor->events & 15u) || (descriptor->read & 63u) ||
        (descriptor->consumed & 3u) || descriptor->events == 0 ||
        descriptor->read == 0 || descriptor->consumed == 0
    ) {
        return 0;
    }
    ring->consumed = (volatile uint32_t*)(uintptr_t)descriptor->consumed;
    ring->events = (volatile uint8_t*)(uintptr_t)descriptor->events;
    ring->read = (volatile uint64_t*)(uintptr_t)descriptor->read;
    ring->command_depth = descriptor->command_depth;
    ring->event_depth = descriptor->event_depth;
    ring->cached = (descriptor->flags & KMX_RING_CACHED) != 0;
    ring->commands = (volatile uint8_t*)(uintptr_t)descriptor->cmd;
    return 1;
}

/* Stores command ring->next: the command's words first, then its number in sequence and
 * the three landing words, so the kernel never sees the number over old words. */
static inline int kmx_ring_submit(
    kmx_ring_t* ring, const km_command_t* command, uint32_t* handle
) {
    if (ring->commands == 0) {
        return KMX_STOPPED;
    }
    const uint32_t number = ring->next;
    if (ring->cached) {
        KMX_RING_SWEEP((uintptr_t)ring->consumed, 4);
    }
    KMX_RING_FENCE_R();
    if ((int32_t)(number - ring->command_depth - *ring->consumed) > 0) {
        return KMX_FULL;
    }
    volatile uint32_t* slot = (
        (volatile uint32_t*)(ring->commands + (number % ring->command_depth) * 64u)
    );
    const uint32_t* words = (const uint32_t*)command;
    for (uint32_t word = 0; word < 16; word++) {
        if (word % 4 != 0) {
            slot[word] = words[word];
        }
    }
    KMX_RING_FENCE_W();
    for (uint32_t word = 0; word < 16; word += 4) {
        slot[word] = number;
    }
    KMX_RING_FENCE_W();
    if (ring->cached) {
        KMX_RING_SWEEP((uintptr_t)slot, 64);
    }
    ring->next = number + 1;
    *handle = number;
    return KMX_OK;
}

/* Takes event ring->taken + 1 if it is whole (its index read before and after its
 * fields), and tells the kernel through the read line. Returns 1 or 0. */
static inline int kmx_ring_take(kmx_ring_t* ring, kmx_event* event) {
    const uint32_t number = ring->taken + 1;
    volatile const uint32_t* line = (
        (volatile const uint32_t*)(ring->events + (number % ring->event_depth) * 16u)
    );
    if (ring->cached) {
        KMX_RING_SWEEP((uintptr_t)line, 16);
    }
    KMX_RING_FENCE_R();
    if (line[0] != number) {
        return 0;
    }
    KMX_RING_FENCE_R();
    const uint32_t request = line[1], status = line[2], ticks = line[3];
    KMX_RING_FENCE_R();
    if (line[0] != number) {
        return 0;
    }
    event->handle = request;
    event->kind = status & 0xFFu;
    event->auxiliary = status >> 8;
    event->ticks = ticks;
    ring->taken = number;
    ring->read[0] = number;
    for (uint32_t word = 1; word < 8; word++) {
        ring->read[word] = 0;
    }
    KMX_RING_FENCE_W();
    if (ring->cached) {
        KMX_RING_SWEEP((uintptr_t)ring->read, 64);
    }
    return 1;
}

/* One event if any: a set-aside one first, else the ring's next. Never waits. */
static inline int kmx_ring_poll(kmx_ring_t* ring, kmx_event* event) {
    if (ring->commands == 0) {
        return KMX_STOPPED;
    }
    if (ring->stashed != 0) {
        *event = ring->stash[0];
        ring->stashed--;
        for (uint32_t position = 0; position < ring->stashed; position++) {
            ring->stash[position] = ring->stash[position + 1];
        }
        return KMX_OK;
    }
    return kmx_ring_take(ring, event) ? KMX_OK : KMX_NONE;
}

/* The event of `handle` if it came (KMX_OK), else KMX_NONE, or KMX_FULL as for
 * kmx_ring_wait; events of other handles are set aside. Never waits. */
static inline int kmx_ring_look(kmx_ring_t* ring, uint32_t handle, kmx_event* event) {
    if (ring->commands == 0) {
        return KMX_STOPPED;
    }
    for (uint32_t position = 0; position < ring->stashed; position++) {
        if (ring->stash[position].handle == handle) {
            *event = ring->stash[position];
            ring->stashed--;
            for (uint32_t later = position; later < ring->stashed; later++) {
                ring->stash[later] = ring->stash[later + 1];
            }
            return KMX_OK;
        }
    }
    kmx_event taken_event;
    while (ring->stashed < KMX_STASH && kmx_ring_take(ring, &taken_event)) {
        if (taken_event.kind == KMX_EVENT_WAKE) {
            continue;
        }
        if (taken_event.handle == handle) {
            *event = taken_event;
            return KMX_OK;
        }
        ring->stash[ring->stashed++] = taken_event;
    }
    return ring->stashed == KMX_STASH ? KMX_FULL : KMX_NONE;
}

/* The event of `handle`, waiting until the timer passes `deadline`; events of other
 * handles are set aside for kmx_poll (KMX_FULL while KMX_STASH are set aside). */
static inline int kmx_ring_wait(
    kmx_ring_t* ring, uint32_t handle, uint64_t deadline, kmx_event* event
) {
    if (ring->commands == 0) {
        return KMX_STOPPED;
    }
    for (uint32_t position = 0; position < ring->stashed; position++) {
        if (ring->stash[position].handle == handle) {
            *event = ring->stash[position];
            ring->stashed--;
            for (uint32_t later = position; later < ring->stashed; later++) {
                ring->stash[later] = ring->stash[later + 1];
            }
            return KMX_OK;
        }
    }
    for (uint32_t spins = 1;; spins++) {
        kmx_event taken_event;
        if (ring->stashed == KMX_STASH) {
            return KMX_FULL; /* nothing is taken that could not be kept */
        }
        if (kmx_ring_take(ring, &taken_event)) {
            if (taken_event.kind == KMX_EVENT_WAKE) {
                continue;
            }
            if (taken_event.handle == handle) {
                *event = taken_event;
                return KMX_OK;
            }
            ring->stash[ring->stashed++] = taken_event;
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
