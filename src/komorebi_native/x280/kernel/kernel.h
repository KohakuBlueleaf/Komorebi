/* The dispatch kernel's state (in the cluster's DRAM bank; the image has no .bss) and
 * the helpers every part uses. One translation unit, main.c, includes the parts. */
#pragma once

#include <stdint.h>

#include "komorebi/abi/km_abi.h"
#include "komorebi/abi/km_stream.h"
#include "km_dispatch.h"
#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"

/* The loop checks stop, deadlines and the idle time every this many passes. */
#define KMD_POLL_EVERY 1024u
/* The team allocator's answer when no team of the count fits now. */
#define KMD_NO_FIRST 0xFFFFFFFFu
/* The loop's hot path stays inlined: an outlined call saves registers on the stack,
 * which is uncached LIM (a load costs about 17 ns). */
#define KMD_HOT static inline __attribute__((always_inline))

/* A command source: its ring of 64-byte command lines, numbered from 1 in slot i %
 * depth, and its ring of 16-byte events; each word has one writer (README.md). */
typedef struct {
    volatile const uint8_t* cmds;  /* the source writes the lines */
    volatile uint32_t* consumed;   /* the kernel writes: commands taken */
    volatile uint8_t* events;      /* the kernel writes the events */
    volatile uint32_t* told;       /* the kernel writes: events pushed (a hint) */
    volatile const uint32_t* read; /* the source writes: events it has read */
    uint32_t depth, event_depth;   /* powers of two */
    uint32_t next;                 /* number of the next command */
    uint32_t pushed;               /* events pushed */
    uint32_t told_n;               /* events the count word shows */
    uint32_t read_n;               /* events the source has read, as last loaded */
    uint32_t line_state; /* 0: line is free; 1: it holds command next; 2: held */
    uint32_t nbusy;      /* busy slots whose record came from here */
    uint32_t owed;       /* events owed besides those (host: one per active stream) */
    uint64_t t_record;   /* deadline of its next record, in ticks */
    uint64_t* line;      /* the command being taken: 8 words, 64-byte aligned */
} kmd_ring_t;

/* Sources, served in this order (the numbers are in km_dispatch.h's hart block). */
#define KMD_SRC_HOST 0u
#define KMD_SRC_ENGINE 1u
#define KMD_SOURCES 2u

/* A stream the kernel admitted and the engine runs: its range and req. */
typedef struct {
    uint32_t active, req, off, bytes;
} kmd_run_t;

typedef struct {
    /* run */
    uint64_t hart, run;
    const volatile kmd_args_t* args;
    volatile uint8_t* lim;          /* the layout's base: the argument block */
    volatile uint32_t* result;      /* the result block */
    volatile uint32_t* harts;       /* the hart block */
    volatile const uint8_t* upload; /* the upload region, upload_bytes long */
    uint32_t upload_bytes;
    uint32_t error, where; /* KMD_ERR_*, KMD_WHERE_* */
    uint64_t t_boot;       /* mtime at the start */
    /* grid port */
    uint32_t panels, opened;  /* windows programmed: bit 0, 1 panels, bit 2 host */
    volatile uint8_t* ctl[2]; /* every core's control block, through each panel */
    /* sources and the host buffer */
    volatile uint32_t* host; /* the host buffer, through the host window */
    kmd_ring_t ring[KMD_SOURCES];
    volatile uint8_t* ering; /* the engine's ring area in DRAM */
    /* completion table */
    uint32_t depth, full, busy, nbusy, cores;
    uint32_t seq;                     /* last sequence number issued */
    uint32_t exited;                  /* an EXIT record was issued */
    uint32_t slot_seq[KM_RING_SLOTS]; /* last sequence: the ack word once free */
    uint32_t slot_req[KM_RING_SLOTS];
    uint32_t slot_team[KM_RING_SLOTS];
    uint8_t slot_src[KM_RING_SLOTS]; /* the source the record came from */
    uint32_t result_busy;            /* busy slots whose record has KM_F_RESULT */
    uint32_t last_result; /* the latest KM_F_RESULT record's result (0: none yet) */
    uint32_t results;     /* KM_F_RESULT records completed */
    uint64_t slot_t0[KM_RING_SLOTS];
    uint64_t slot_deadline[KM_RING_SLOTS];
    /* streams the engine runs */
    kmd_run_t runs[KM_STREAM_MAX_ACTIVE];
    uint32_t nstreams; /* admitted and not yet ended */
    /* counters */
    uint32_t records, max_busy, refused, held, no_room;
    uint32_t checked, bad_lines, bad_first; /* KMD_VERIFY_LINES builds */
    uint32_t traced, untraced;              /* KMD_TRACE builds */
    uint64_t passes, idle_passes, full_passes;
    uint64_t t_first, t_last_event;
} kmd_t;

_Static_assert(sizeof(kmd_t) <= KMD_STATE_ENGINE, "kmd_t outgrows its DRAM area");

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
