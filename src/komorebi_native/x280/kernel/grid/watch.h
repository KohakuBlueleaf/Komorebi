/* Core 2 watches core 0's loop: a stall is reported to the host, then core 0 is sent
 * a machine software interrupt, whose handler (trap.h) re-enters the loop. */
#pragma once

#include "../kernel.h"
#include "../sources/ring.h"

/* Hart `hart`'s machine software interrupt pending bit in the cluster's CLINT. */
static inline volatile uint32_t* watch_msip(uint32_t hart) {
    return (volatile uint32_t*)(uintptr_t)(X280_MSIP_BASE + 4u * hart);
}

/* Core 0, as its loop starts: the watch on, then its software interrupt enabled (the
 * only one). */
static inline void watch_on(kmd_t* dispatcher) {
    dispatcher->x280_cores[KMD_H_WATCH] = KMD_WATCH_ON;
    x280_fence_ow();
    __asm__ volatile("csrw mie, %0" ::"r"(1ul << 3));
    __asm__ volatile("csrs mstatus, %0" ::"r"(1ul << 3));
}

/* Core 0, as its loop ends (or it traps): interrupts off, then the watch off. */
static inline void watch_off(volatile uint32_t* x280_cores) {
    __asm__ volatile("csrc mstatus, %0" ::"r"(1ul << 3));
    __asm__ volatile("csrc mie, %0" ::"r"(1ul << 3));
    x280_cores[KMD_H_WATCH] = KMD_WATCH_OFF;
    x280_fence_ow();
}

/* Core 0's heartbeat: its passes plus the beats of its long waits. */
static inline uint64_t watch_beat(const kmd_t* dispatcher) {
    const volatile uint64_t* passes = &dispatcher->passes;
    const volatile uint64_t* beat = &dispatcher->beat;
    return *passes + *beat;
}

/* An event on core 0's host ring while core 0 is stalled; none without room. */
static void watch_push(
    kmd_t* dispatcher, uint32_t request, uint32_t stage, uint32_t ticks
) {
    __asm__ volatile("" ::: "memory");
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_HOST];
    if (ring->events == 0 || !ring_room(ring)) {
        return;
    }
    ring_push(ring, request, KMD_EVENT_STALL | stage << 8, ticks);
    ring_flush(ring);
}

/* One stall: SEEN, the interrupt, then core 0's re-entry awaited. 1: it came back. */
static int watch_stall(kmd_t* dispatcher, uint32_t gap) {
    volatile uint32_t* x280_cores = dispatcher->x280_cores;
    const uint32_t back = x280_cores[KMD_H_BACK];
    const uint64_t seen = x280_mtime();
    x280_cores[KMD_H_STALLS]++;
    x280_cores[KMD_H_STALL_PASSES] = (uint32_t)dispatcher->passes;
    *(volatile uint64_t*)&dispatcher->stall_mtime = seen;
    watch_push(dispatcher, dispatcher->ring[KMD_SRC_HOST].next, KMD_STALL_SEEN, gap);
    *watch_msip(0) = 1u;
    x280_fence_iorw();
    while (x280_mtime() - seen < KMD_WATCH_BACK_TICKS) {
        if (x280_cores[KMD_H_BACK] != back || x280_cores[KMD_H_WATCH] != KMD_WATCH_ON) {
            return 1;
        }
    }
    *watch_msip(0) = 0u;
    x280_fence_iorw();
    if (x280_cores[KMD_H_WATCH] != KMD_WATCH_ON) {
        return 1; /* its loop ended meanwhile */
    }
    x280_cores[KMD_H_WATCH] = KMD_WATCH_LOST;
    watch_push(
        dispatcher,
        dispatcher->ring[KMD_SRC_HOST].next,
        KMD_STALL_LOST,
        (uint32_t)(x280_mtime() - seen)
    );
    return 0;
}

/* Core 2's run: waits for core 0's loop, then looks at its heartbeat every
 * KMD_WATCH_POLL_TICKS until the loop ends, a stall is lost or the runtime stops. */
static void watch(uint64_t core, const volatile kmd_args_t* arguments, uint64_t run) {
    kmd_t* dispatcher = (
        (kmd_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + arguments->state_ddr)
    );
    volatile uint32_t* x280_cores = (
        (volatile uint32_t*)((volatile uint8_t*)arguments + KMD_LIM_CORES)
    );
    uint64_t last = 0, since = 0, gap = 0;
    while (!x280_hart_stop_requested(core, run)) {
        const uint32_t state = x280_cores[KMD_H_WATCH];
        if (state != KMD_WATCH_ON) {
            if (state == 0) {
                continue; /* core 0 has not reached its loop */
            }
            break;
        }
        const uint64_t now = x280_mtime();
        const uint64_t beat = watch_beat(dispatcher);
        if (since == 0 || beat != last) {
            gap = since != 0 && now - since > gap ? now - since : gap;
            last = beat;
            since = now;
        } else if (now - since > KMD_WATCH_STALL_TICKS) {
            gap = now - since > gap ? now - since : gap;
            if (!watch_stall(dispatcher, (uint32_t)(now - since))) {
                break;
            }
            since = x280_mtime();
            last = watch_beat(dispatcher);
        }
        x280_cores[KMD_H_WATCH_GAP] = (uint32_t)gap;
        while (x280_mtime() - now < KMD_WATCH_POLL_TICKS) {
            ;
        }
    }
}
