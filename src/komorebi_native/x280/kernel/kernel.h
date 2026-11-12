/* The dispatch kernel's state and the helpers every part uses. One translation unit
 * (main.c) includes the parts; the image has no .bss, so the state lives on the stack.
 */
#pragma once

#include <stdint.h>

#include "komorebi/abi/km_abi.h"
#include "km_dispatch.h"
#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"

/* The loop checks stop, deadlines and the idle time every this many passes. */
#define KMD_POLL_EVERY 1024u

typedef struct {
    /* run */
    uint64_t hart, run;
    const volatile kmd_args_t* args;
    volatile uint8_t* lim;     /* the layout's base: the argument block */
    volatile uint32_t* result; /* the result block */
    uint32_t error, where;     /* KMD_ERR_*, KMD_WHERE_* */
    uint64_t t_boot;           /* mtime at the start */
    /* grid port */
    uint32_t panels, opened;  /* windows programmed: bit 0, 1 panels, bit 2 host */
    volatile uint8_t* ctl[2]; /* every core's control block, through each panel */
    /* host ring and events */
    volatile uint32_t* host; /* the host buffer, through the host window */
    uint32_t host_depth, event_depth;
    uint32_t next;       /* number of the next host command */
    uint32_t events;     /* events pushed */
    uint32_t told;       /* events the host's count word shows */
    uint32_t host_read;  /* events the host has read, as last loaded from LIM */
    uint32_t line_state; /* 0: k->line is free; 1: it holds command k->next; 2: held */
    /* completion table */
    uint32_t depth, full, busy, nbusy, cores;
    uint32_t seq;                     /* last sequence number issued */
    uint32_t exited;                  /* an EXIT record was issued */
    uint64_t t_record;                /* deadline of the next record, in ticks */
    uint32_t slot_seq[KM_RING_SLOTS]; /* last sequence: the ack word once free */
    uint32_t slot_req[KM_RING_SLOTS];
    uint32_t slot_team[KM_RING_SLOTS];
    uint64_t slot_t0[KM_RING_SLOTS];
    uint64_t slot_deadline[KM_RING_SLOTS];
    /* counters */
    uint32_t records, max_busy, refused, held, no_room;
    uint64_t passes, idle_passes, full_passes;
    uint64_t t_first, t_last_event;
    /* the command being taken, as 8 little-endian 64-bit words, on the stack and
     * 64-byte aligned for the vector store */
    uint64_t* line;
} kmd_t;

_Static_assert(sizeof(kmd_t) <= KMD_STATE_BYTES, "kmd_t outgrows its DRAM area");

/* take()'s outcome: the command was consumed, waits in k->line, or ends the loop. */
typedef enum {
    KMD_TAKEN = 0,
    KMD_HELD = 1,
    KMD_NO_ROOM = 2,
    KMD_FINISH = 3
} kmd_take_t;

/* A wait's outcome. */
typedef enum { KMD_OK = 0, KMD_STOP = 1, KMD_LATE = 2 } kmd_wait_t;

/* Every KMD_POLL_EVERY calls: whether the runtime asked for a stop or `deadline`
 * (mtime, 0 = none) has passed. */
static inline kmd_wait_t kmd_poll(const kmd_t* k, uint32_t* spins, uint64_t deadline) {
    if (++*spins != KMD_POLL_EVERY) {
        return KMD_OK;
    }
    *spins = 0;
    if (x280_hart_stop_requested(k->hart, k->run)) {
        return KMD_STOP;
    }
    if (deadline != 0 && x280_mtime() > deadline) {
        return KMD_LATE;
    }
    return KMD_OK;
}

/* Stores the 64-byte line at `src` to `dst` with one vector store (VLEN 512 = 16 x
 * e32); both 64-byte aligned. */
static inline void kmd_store_line(volatile void* dst, volatile const void* src) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 16, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(src), "r"(dst)
        : "memory", "v8"
    );
}

/* Stores the 16-byte chunk at `src` to `dst` with one vector store. */
static inline void kmd_store_chunk(volatile void* dst, const uint32_t* src) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v9, (%0)\n"
        "vse32.v v9, (%1)\n"
        ".option pop\n"
        :
        : "r"(src), "r"(dst)
        : "memory", "v9"
    );
}
