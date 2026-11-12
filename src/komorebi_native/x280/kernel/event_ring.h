/* The host buffer: a posted window onto it through the PCIe tile; the host's event
 * ring and consumed count in it, the state words, the ready block. */
#pragma once

#include "kernel.h"
#include "ring.h"

/* Programs the host window and points the host's rings at it. Returns KMD_ERR_NONE or
 * KMD_ERR_WINDOW. */
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
    kmd_ring_t* r = &k->ring[KMD_SRC_HOST];
    r->consumed = &k->host[KMD_HOST_CONSUMED];
    r->events = (volatile uint8_t*)k->host + KMD_HOST_RING;
    r->told = &k->host[KMD_HOST_EVENTS];
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

/* An event to the host. */
static inline void host_push(kmd_t* k, uint32_t req, uint32_t status, uint32_t ticks) {
    ring_push(&k->ring[KMD_SRC_HOST], req, status, ticks);
}

/* Clears the host window if it was programmed. */
static inline void events_close(kmd_t* k) {
    x280_fence_ow();
    if (k->opened & (1u << 2)) {
        noc_clear_tlb_2m(KMD_WIN_HOST);
        k->opened &= ~(1u << 2);
        k->host = 0;
        k->ring[KMD_SRC_HOST].events = 0;
    }
}
