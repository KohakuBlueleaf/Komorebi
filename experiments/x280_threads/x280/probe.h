/* Shared by the probe images: the LIM layout the host writes and reads, the argument
 * check, progress markers, timed waits and the cache operations under test. */
#pragma once

#include <stdint.h>

#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"

/* LIM layout, in bytes from the cluster's available LIM base (all 64-byte lines). */
#define PR_ARGS 0x000u /* argument block: run value, then the probe's own words */
#define PR_MARKER                                                                      \
    0x100u               /* word h: core h's last progress marker, word 4 + h its time \
                          */
#define PR_GO 0x140u     /* line from the host: word 0 = the step it allows */
#define PR_RESULT 0x200u /* the probe's result words */
#define PR_RESULT_WORDS 256u
#define PR_BYTES 0x600u

/* Core results: 0 for a core with nothing to do, else one of these or a count. */
#define PR_ERR_ARGS 0xE0u  /* the argument block is not this run's or out of range */
#define PR_ERR_LATE 0xE1u  /* a timed wait ran out */
#define PR_ERR_STOP 0xE2u  /* the runtime asked for a stop */
#define PR_ERR_FAULT 0xE3u /* a probed instruction faulted (and was skipped) */

/* The cluster's L2 cache controller: Flush64 writes back and invalidates the line
 * holding the physical address stored into it (rt/x280_pinned_input.h). */
#define PR_CCACHE_FLUSH64 0x02010200ull

typedef struct {
    uint64_t core, run;
    volatile uint8_t* lim;
    uint64_t deadline; /* mtime; every wait ends by it */
} pr_t;

static inline volatile uint32_t* pr_word(const pr_t* p, uint32_t offset) {
    return (volatile uint32_t*)(p->lim + offset);
}

/* Floating point and vector on: entry leaves FS off and the first FP use traps. */
static inline void pr_enable_fpv(void) {
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
}

/* Reads this run's argument block; 0 when it is not this run's or too short. */
static inline int pr_open(pr_t* p, uint64_t core, uint64_t min_bytes) {
    p->core = core;
    p->run = x280_hart_control(core)->run_value;
    p->lim = (volatile uint8_t*)(uintptr_t)x280_hart_args(core);
    p->deadline = 0;
    if (p->lim == 0 || x280_hart_args_bytes() < min_bytes) {
        return 0;
    }
    return *(volatile uint64_t*)p->lim == p->run;
}

static inline void pr_marker(const pr_t* p, uint32_t where) {
    pr_word(p, PR_MARKER)[p->core] = where;
    pr_word(p, PR_MARKER)[4 + p->core] = (uint32_t)x280_mtime();
    x280_fence_ow();
}

/* 0 while the run may go on, else PR_ERR_STOP or PR_ERR_LATE. */
static inline uint32_t pr_check(const pr_t* p) {
    if (x280_hart_stop_requested(p->core, p->run)) {
        return PR_ERR_STOP;
    }
    if (x280_mtime() > p->deadline) {
        return PR_ERR_LATE;
    }
    return 0;
}

/* Waits until the host's go line holds `step` (or more); 0 or an error. */
static inline uint32_t pr_wait_go(const pr_t* p, uint32_t step) {
    volatile const uint32_t* go = pr_word(p, PR_GO);
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (*go >= step) {
            return 0;
        }
        if ((spins & 255u) == 0) {
            const uint32_t e = pr_check(p);
            if (e) {
                return e;
            }
        }
    }
}

/* Flush64 of every line of [addr, addr + bytes). The store is a 4-byte instruction
 * (no compressed form), so probe mode can skip it if it faults. */
static inline void pr_flush(uint64_t addr, uint64_t bytes) {
    volatile uint64_t* reg = (volatile uint64_t*)(uintptr_t)PR_CCACHE_FLUSH64;
    for (uint64_t a = addr & ~63ull; a < addr + bytes; a += 64) {
        __asm__ volatile(".option push\n.option norvc\nsd %0, 0(%1)\n.option pop\n"
                         :
                         : "r"(a), "r"(reg)
                         : "memory");
    }
    x280_fence_iorw();
}

/* Arms probe mode on this core: an access or illegal-instruction fault is skipped
 * and noted instead of halting the core. pr_unprobe says whether one happened. */
static inline void pr_probe(uint64_t core) {
    *x280_hart_probe(core) = 1;
    x280_fence_iorw();
}

static inline int pr_unprobe(uint64_t core) {
    x280_fence_iorw();
    const int faulted = *x280_hart_probe(core) == 2;
    *x280_hart_probe(core) = 0;
    x280_fence_iorw();
    return faulted;
}
