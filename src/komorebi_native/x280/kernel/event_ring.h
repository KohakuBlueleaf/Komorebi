/* Event ring to host memory: a posted window onto the host buffer through the PCIe
 * tile; 16-byte events, the state words and the consumed count. */
#pragma once

#include "kernel.h"

/* Programs the host window. Returns KMD_ERR_NONE or KMD_ERR_WINDOW. */
static inline uint32_t events_open(kmd_t* k) {
    const volatile kmd_args_t* a = k->args;
    noc_tlb_2m_t programmed;
    volatile void* win = (
        noc_configure_tlb_2m_checked(
            KMD_WIN_HOST,
            (uint32_t)a->pcie_x,
            (uint32_t)a->pcie_y,
            a->host_noc_addr,
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (win == 0) {
        return KMD_ERR_WINDOW;
    }
    k->opened |= 1u << 2;
    if (noc_verify_tlb_2m(KMD_WIN_HOST, &programmed) != 0) {
        return KMD_ERR_WINDOW;
    }
    k->host = (volatile uint32_t*)win;
    return KMD_ERR_NONE;
}

/* Stores one state word of the host buffer. */
static inline void events_word(kmd_t* k, uint32_t word, uint32_t value) {
    if (k->host != 0) {
        k->host[word] = value;
        x280_fence_ow();
    }
}

/* Stores the ready block (kmd_ready_t) as one 16-byte store. */
static inline void events_ready(kmd_t* k) {
    if (k->host == 0) {
        return;
    }
    uint32_t r[4] __attribute__((aligned(16)));
    r[0] = KMD_READY_MARK;
    r[1] = (uint32_t)(uintptr_t)k->upload;
    r[2] = k->upload_bytes;
    r[3] = KMD_LIM_BYTES;
    kmd_store_chunk((volatile uint8_t*)k->host + KMD_HOST_READY * 4u, r);
    x280_fence_ow();
}

/* Stores event ++events {req, status, ticks} as one 16-byte store (README.md: one store
 * per event); events_flush makes it known. */
static inline void events_push(
    kmd_t* k, uint32_t req, uint32_t status, uint32_t ticks
) {
    if (k->host == 0) {
        return;
    }
    const uint32_t index = ++k->events;
    uint32_t ev[4] __attribute__((aligned(16)));
    ev[0] = index;
    ev[1] = req;
    ev[2] = status;
    ev[3] = ticks;
    volatile uint8_t* ring = (volatile uint8_t*)k->host + KMD_HOST_RING;
    kmd_store_chunk(ring + (index % k->event_depth) * sizeof(km_event_t), ev);
}

/* Whether the unread events, one per busy slot and one more fit the ring. The host's
 * read count is loaded from LIM only when the cached one says no. */
static inline int events_room(kmd_t* k) {
    if (k->events - k->host_read + k->nbusy < k->event_depth) {
        return 1;
    }
    x280_fence_ir();
    k->host_read = *(volatile const uint32_t*)(k->lim + KMD_LIM_READ);
    return k->events - k->host_read + k->nbusy < k->event_depth;
}

/* Once per loop pass after events were pushed: one fence, then the event count (a
 * hint for the host). */
static inline void events_flush(kmd_t* k) {
    if (k->events != k->told && k->host != 0) {
        x280_fence_ow();
        k->host[KMD_HOST_EVENTS] = k->events;
        k->told = k->events;
    }
}

/* Clears the host window if it was programmed. */
static inline void events_close(kmd_t* k) {
    x280_fence_ow();
    if (k->opened & (1u << 2)) {
        noc_clear_tlb_2m(KMD_WIN_HOST);
        k->opened &= ~(1u << 2);
        k->host = 0;
    }
}
