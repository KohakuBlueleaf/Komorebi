/* The host buffer: a posted window onto it through the PCIe tile; the host's event
 * ring and consumed count in it, the state words, the ready block. */
#pragma once

#include "kernel.h"
#include "ring.h"

/* Programs the host window and points the host's rings at it. Returns KMD_ERROR_NONE or
 * KMD_ERROR_WINDOW. */
static inline uint32_t events_open(kmd_t* dispatcher) {
    const volatile kmd_args_t* arguments = dispatcher->args;
    noc_tlb_2m_t programmed;
    volatile void* window = (
        noc_configure_tlb_2m_checked(
            KMD_WIN_HOST,
            (uint32_t)arguments->pcie_x,
            (uint32_t)arguments->pcie_y,
            arguments->host_noc_addr,
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (window == 0) {
        return KMD_ERROR_WINDOW;
    }
    dispatcher->opened |= 1u << 2;
    if (noc_verify_tlb_2m(KMD_WIN_HOST, &programmed) != 0) {
        return KMD_ERROR_WINDOW;
    }
    dispatcher->host = (volatile uint32_t*)window;
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_HOST];
    ring->consumed = &dispatcher->host[KMD_HOST_CONSUMED];
    ring->events = (volatile uint8_t*)dispatcher->host + KMD_HOST_RING;
    ring->told = &dispatcher->host[KMD_HOST_EVENTS];
    return KMD_ERROR_NONE;
}

/* Stores one state word of the host buffer. */
static inline void events_word(kmd_t* dispatcher, uint32_t word, uint32_t value) {
    if (dispatcher->host != 0) {
        dispatcher->host[word] = value;
        x280_fence_ow();
    }
}

/* Stores the ready block (kmd_ready_t) as one 16-byte store. */
static inline void events_ready(kmd_t* dispatcher) {
    if (dispatcher->host == 0) {
        return;
    }
    uint32_t words[4] __attribute__((aligned(16)));
    words[0] = KMD_READY_MARK;
    words[1] = (uint32_t)(uintptr_t)dispatcher->upload;
    words[2] = dispatcher->upload_bytes;
    words[3] = KMD_LIM_BYTES;
    kmd_store_chunk((volatile uint8_t*)dispatcher->host + KMD_HOST_READY * 4u, words);
    x280_fence_ow();
}

/* An event to the host. */
static inline void host_push(
    kmd_t* dispatcher, uint32_t request, uint32_t status, uint32_t ticks
) {
    ring_push(&dispatcher->ring[KMD_SRC_HOST], request, status, ticks);
}

/* Clears the host window if it was programmed. */
static inline void events_close(kmd_t* dispatcher) {
    x280_fence_ow();
    if (dispatcher->opened & (1u << 2)) {
        noc_clear_tlb_2m(KMD_WIN_HOST);
        dispatcher->opened &= ~(1u << 2);
        dispatcher->host = 0;
        dispatcher->ring[KMD_SRC_HOST].events = 0;
    }
}
