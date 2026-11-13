/* The dispatcher's state (in the cluster's DRAM bank; the image has no .bss) and
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

/* The L3 cache's flush register: a store of a physical line address writes the line
 * back and drops it. */
#define KMD_CCACHE_FLUSH64 0x02010200ull
/* The loop checks stop, deadlines and the idle time every this many passes. */
#define KMD_POLL_EVERY 1024u
/* The group allocator's answer when no group of the count fits now. */
#define KMD_NO_FIRST 0xFFFFFFFFu
/* The loop's hot path stays inlined: an outlined call saves registers on the stack,
 * which is uncached LIM (a load costs about 17 ns). */
#define KMD_HOT static inline __attribute__((always_inline))

/* A command source: its ring of 64-byte command lines, numbered from 1 in slot i %
 * depth, and its ring of 16-byte events; each word has one writer (README.md). */
typedef struct {
    volatile const uint8_t* commands; /* the source writes the lines */
    volatile uint32_t* consumed;      /* the kernel writes: commands taken */
    volatile uint8_t* events;         /* the kernel writes the events */
    volatile uint32_t* told;          /* the kernel writes: events pushed (a hint) */
    volatile const uint32_t* read;    /* the source writes: events it has read */
    uint32_t depth, event_depth;      /* powers of two */
    uint32_t next;                    /* number of the next command */
    uint32_t pushed;                  /* events pushed */
    uint32_t told_count;              /* events the count word shows */
    uint32_t read_count;              /* events the source has read, as last loaded */
    uint32_t line_state;   /* 0: line is free; 1: it holds command next; 2: held */
    uint32_t busy_count;   /* busy slots whose command came from here */
    uint32_t queued_count; /* its jobs waiting in the queue */
    uint32_t owed;         /* events owed besides those (host: one per active stream) */
    uint64_t t_command;    /* deadline of its next command, in ticks */
    uint64_t before;       /* queue places of jobs queued before the line was taken */
    uint64_t* line;        /* the command being taken: 8 words, 64-byte aligned */
} kmd_ring_t;

/* Sources, served in this order (the numbers are in km_dispatch.h's core block). */
#define KMD_SRC_HOST 0u
#define KMD_SRC_ENGINE 1u
/* work cluster c's thread: KMD_SRC_WORK + c - 1 (work_port.h) */
#define KMD_SRC_WORK 2u
#define KMD_SOURCES (KMD_SRC_WORK + KMD_WORK_CLUSTERS)

/* A job waiting in the queue: its command, with its source's deadline at the time. */
typedef struct {
    uint64_t line[8];
    uint64_t ticks;
    uint32_t request, count,
        passed; /* passed: jobs behind it submitted while it waited */
    uint8_t source, pad;
    uint16_t tag; /* the engine's stream tag (KMD_HANDLER_DROP) */
} kmd_job_t;

/* The job queue (KMD_STATE_QUEUE): jobs in places, their order oldest first. */
typedef struct {
    kmd_job_t job[KM_QUEUE_JOBS];
    uint64_t used;   /* a bit per place */
    uint32_t length; /* jobs waiting */
    uint32_t taken;  /* the engine's jobs taken in, ever */
    uint8_t order[KM_QUEUE_JOBS];
} kmd_queue_t;

/* The dynamic run in flight (dynamic.h): its RUN command, and its end command's slot.
 */
typedef struct {
    uint32_t handler; /* the program's dynamic handlers, one per byte; 0: none */
    uint32_t active;  /* a RUN is in flight and not yet quiet */
    uint32_t slot, sequence, group, request, source; /* the RUN command */
    uint32_t end; /* 1 + the end command's slot while busy; 0 */
    uint32_t checks;
    uint32_t first[2u * KM_CORES_MAX]; /* the first pass: sent, consumed per core */
} kmd_messages_t;

/* Commands handed to a work cluster's thread (work_port.h), line n in slot n % depth;
 * stored only once the thread said it serves, as its start-up writes the area over. */
typedef struct {
    volatile uint8_t* lines; /* the thread's command lines, through the ring's window */
    uint32_t next;           /* number of the next line, from 1 */
    uint32_t serving;        /* the thread's answer to line 0 came: it serves */
    uint32_t unstored;       /* bit k: slot k's line waits for that */
    uint32_t busy;           /* bit k: slot k's line is not answered yet */
    uint32_t late;   /* bit k: its timeout event went out; the answer sends none */
    uint32_t result; /* bit k: its command has KM_FLAG_RESULT */
    uint32_t number[KMD_WORK_IN_DEPTH];
    uint32_t request[KMD_WORK_IN_DEPTH];
    uint32_t sequence[KMD_WORK_IN_DEPTH];
    uint8_t source[KMD_WORK_IN_DEPTH];
    uint64_t submitted[KMD_WORK_IN_DEPTH];
    uint64_t deadline[KMD_WORK_IN_DEPTH];
    uint64_t line[KMD_WORK_IN_DEPTH][8]; /* each line as the thread is to see it */
} kmd_thread_t;

/* A stream the kernel admitted and the engine runs: its range and request. */
typedef struct {
    uint32_t active, request, offset, bytes;
} kmd_run_t;

typedef struct {
    /* run */
    uint64_t core, run;
    const volatile kmd_args_t* args;
    volatile uint8_t* lim;          /* the layout's base: the argument block */
    volatile uint32_t* result;      /* the result block */
    volatile uint32_t* x280_cores;  /* the core block */
    volatile const uint8_t* upload; /* the upload region, upload_bytes long */
    uint32_t upload_bytes;
    uint32_t error, where; /* KMD_ERROR_*, KMD_WHERE_* */
    uint64_t start_mtime;  /* mtime at the start */
    /* grid port */
    uint32_t panels, opened; /* windows programmed: bits 0, 1 panels, 2 host, 3+ work */
    volatile uint8_t* control[2]; /* every core's control block, through each panel */
    /* sources and the host buffer */
    volatile uint32_t* host; /* the host buffer, through the host window */
    kmd_ring_t ring[KMD_SOURCES];
    volatile uint8_t* ering; /* the engine's ring area in DRAM */
    kmd_queue_t* queue;      /* the job queue, in the state area */
    /* completion table */
    uint32_t depth, full, busy, busy_count, cores;
    uint32_t sequence;                     /* last sequence number submitted */
    uint32_t exited;                       /* an EXIT command was submitted */
    uint32_t slot_sequence[KM_RING_SLOTS]; /* last sequence: the ack word once free */
    uint32_t slot_request[KM_RING_SLOTS];
    uint32_t slot_group[KM_RING_SLOTS];
    uint8_t slot_source[KM_RING_SLOTS]; /* the source the command came from */
    uint32_t result_busy; /* busy slots whose command has KM_FLAG_RESULT */
    uint32_t last_result; /* the latest KM_FLAG_RESULT command's result (0: none yet) */
    uint32_t results;     /* KM_FLAG_RESULT commands completed */
    uint64_t slot_submitted[KM_RING_SLOTS];
    uint64_t slot_deadline[KM_RING_SLOTS];
    /* streams the engine runs */
    kmd_run_t runs[KM_STREAM_MAX_ACTIVE];
    uint32_t stream_count; /* admitted and not yet ended */
    uint32_t engine_on;    /* engine_active() after its last turn; a RUN sets it */
    uint32_t queued;       /* the queue's length, for the loop */
    uint32_t waits;        /* bit source: a busy rejection to it, KMD_EVENT_WAKE owed */
    kmd_messages_t messages;
    kmd_thread_t thread[KMD_WORK_CLUSTERS]; /* commands to work cluster c at c - 1 */
    uint32_t thread_busy; /* lines handed to threads and not answered, all clusters */
    /* counters */
    uint32_t commands, max_busy, rejected, held, no_room;
    uint32_t checked, bad_lines, bad_first; /* KMD_VERIFY_LINES builds */
    uint32_t traced, untraced;              /* KMD_TRACE builds */
    uint64_t passes, idle_passes, full_passes;
    uint64_t first_command_mtime, last_event_mtime;
} kmd_t;

_Static_assert(sizeof(kmd_t) <= KMD_STATE_ENGINE, "kmd_t outgrows its DRAM area");
_Static_assert(
    KMD_STATE_QUEUE + sizeof(kmd_queue_t) <= KMD_STATE_BYTES,
    "the queue outgrows the state"
);
_Static_assert(KM_QUEUE_JOBS <= 64u, "a 64-bit map of the queue's places");

/* take()'s outcome: the command was consumed, waits in its ring's line, or ends the
 * loop. */
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
static inline kmd_wait_t kmd_poll(
    const kmd_t* dispatcher, uint32_t* spins, uint64_t deadline
) {
    if (++*spins != KMD_POLL_EVERY) {
        return KMD_OK;
    }
    *spins = 0;
    if (x280_hart_stop_requested(dispatcher->core, dispatcher->run)) {
        return KMD_STOP;
    }
    if (deadline != 0 && x280_mtime() > deadline) {
        return KMD_LATE;
    }
    return KMD_OK;
}

/* Stores the 64-byte line at `source` to `destination` with one vector store (VLEN 512
 * = 16 x e32); both 64-byte aligned. */
static inline void kmd_store_line(
    volatile void* destination, volatile const void* source
) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 16, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(source), "r"(destination)
        : "memory", "v8"
    );
}

/* Stores the 16-byte chunk at `source` to `destination` with one vector store. */
static inline void kmd_store_chunk(volatile void* destination, const uint32_t* source) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v9, (%0)\n"
        "vse32.v v9, (%1)\n"
        ".option pop\n"
        :
        : "r"(source), "r"(destination)
        : "memory", "v9"
    );
}

/* Writes the lines of [from, from + bytes) back to DRAM and drops them from the cache,
 * so no dirty line is written over the next process's data later. */
static inline void kmd_flush_lines(const volatile void* from, uint64_t bytes) {
    volatile uint64_t* flush = (volatile uint64_t*)(uintptr_t)KMD_CCACHE_FLUSH64;
    for (uint64_t offset = 0; offset < bytes; offset += 64u) {
        *flush = (uint64_t)(uintptr_t)from + offset;
        x280_fence_iorw();
    }
}
