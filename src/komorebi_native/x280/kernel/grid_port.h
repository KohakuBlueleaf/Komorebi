/* Grid port: the panel windows onto every core's control block, the record store and
 * the stop word. The kernel only writes to the grid; it never reads it. */
#pragma once

#include "kernel.h"

/* Programs one posted window per panel aimed at the control block `ctl` of every core
 * in the panel's rectangle. Returns KMD_ERR_NONE or KMD_ERR_WINDOW. */
static inline uint32_t grid_open(kmd_t* k, uint32_t ctl) {
    for (uint32_t r = 0; r < k->panels; r++) {
        const volatile uint32_t* q = k->args->rect[r];
        const uint32_t x0 = q[0], y0 = q[1], x1 = q[2], y1 = q[3];
        const int mcast = !(x0 == x1 && y0 == y1);
        noc_tlb_2m_t programmed;
        volatile void* win = (
            noc_configure_tlb_2m_mcast(
                KMD_WIN_PANEL0 + r,
                x0,
                y0,
                x1,
                y1,
                0,
                NOC_ORDER_POSTED,
                mcast,
                0u,
                &programmed
            )
        );
        if (win == 0) {
            return KMD_ERR_WINDOW;
        }
        k->opened |= 1u << r;
        if (noc_verify_tlb_2m(KMD_WIN_PANEL0 + r, &programmed) != 0) {
            return KMD_ERR_WINDOW;
        }
        k->ctl[r] = (volatile uint8_t*)win + ctl;
    }
    return KMD_ERR_NONE;
}

/* Stores the record `line` into mailbox slot `slot` of every core: one 64-byte vector
 * store per panel, then a fence so the stores leave now. */
KMD_HOT void grid_store_record(kmd_t* k, uint32_t slot, const uint64_t* line) {
    const uint32_t off = KM_CTL_RING + slot * (uint32_t)sizeof(km_record_t);
    for (uint32_t r = 0; r < k->panels; r++) {
        kmd_store_line(k->ctl[r] + off, line);
    }
    x280_fence_ow();
}

/* Stores a nonzero stop word into every core's control block through each panel that
 * opened: every wait on the grid gives up. */
static inline void grid_stop(kmd_t* k) {
    for (uint32_t r = 0; r < k->panels; r++) {
        if (k->ctl[r] != 0) {
            *(volatile uint32_t*)(k->ctl[r] + KM_CTL_STOP) = 1u;
        }
    }
    x280_fence_ow();
}

/* Clears the panel windows that were programmed. */
static inline void grid_close(kmd_t* k) {
    x280_fence_ow();
    for (uint32_t r = 0; r < 2; r++) {
        if (k->opened & (1u << r)) {
            noc_clear_tlb_2m(KMD_WIN_PANEL0 + r);
            k->opened &= ~(1u << r);
        }
    }
}
