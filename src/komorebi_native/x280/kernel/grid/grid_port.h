/* Grid port: the panel windows onto every core's control block, the command store and
 * the stop flag. The kernel only writes to the grid; it never reads it. */
#pragma once

#include "../kernel.h"

/* Programs one posted window per panel aimed at the control block `control` of every
 * core in the panel's rectangle. Returns KMD_ERROR_NONE or KMD_ERROR_WINDOW. */
static inline uint32_t grid_open(kmd_t* dispatcher, uint32_t control) {
    for (uint32_t panel = 0; panel < dispatcher->panels; panel++) {
        const volatile uint32_t* rectangle = dispatcher->args->rect[panel];
        const uint32_t x_start = rectangle[0];
        const uint32_t y_start = rectangle[1];
        const uint32_t x_end = rectangle[2];
        const uint32_t y_end = rectangle[3];
        const int multicast = !(x_start == x_end && y_start == y_end);
        noc_tlb_2m_t programmed;
        volatile void* window = (
            noc_configure_tlb_2m_mcast(
                KMD_WIN_PANEL0 + panel,
                x_start,
                y_start,
                x_end,
                y_end,
                0,
                NOC_ORDER_POSTED,
                multicast,
                0u,
                &programmed
            )
        );
        if (window == 0) {
            return KMD_ERROR_WINDOW;
        }
        dispatcher->opened |= 1u << panel;
        if (noc_verify_tlb_2m(KMD_WIN_PANEL0 + panel, &programmed) != 0) {
            return KMD_ERROR_WINDOW;
        }
        dispatcher->control[panel] = (volatile uint8_t*)window + control;
    }
    return KMD_ERROR_NONE;
}

/* Stores the command `line` into mailbox slot `slot` of every core: one 64-byte vector
 * store per panel, then a fence so the stores leave now. */
KMD_HOT void grid_store_command(
    kmd_t* dispatcher, uint32_t slot, const uint64_t* line
) {
    const uint32_t offset = KM_CONTROL_RING + slot * (uint32_t)sizeof(km_command_t);
    for (uint32_t panel = 0; panel < dispatcher->panels; panel++) {
        kmd_store_line(dispatcher->control[panel] + offset, line);
    }
    x280_fence_ow();
}

/* Stores a nonzero stop flag into every core's control block through each panel that
 * opened: every wait on the grid gives up. */
static inline void grid_stop(kmd_t* dispatcher) {
    for (uint32_t panel = 0; panel < dispatcher->panels; panel++) {
        if (dispatcher->control[panel] != 0) {
            *(volatile uint32_t*)(dispatcher->control[panel] + KM_CONTROL_STOP) = 1u;
        }
    }
    x280_fence_ow();
}

/* Clears the panel windows that were programmed. */
static inline void grid_close(kmd_t* dispatcher) {
    x280_fence_ow();
    for (uint32_t panel = 0; panel < 2; panel++) {
        if (dispatcher->opened & (1u << panel)) {
            noc_clear_tlb_2m(KMD_WIN_PANEL0 + panel);
            dispatcher->opened &= ~(1u << panel);
        }
    }
}
