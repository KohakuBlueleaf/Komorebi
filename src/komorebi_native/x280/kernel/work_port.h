/* Core 0's side of a work cluster's ring (source 2): commands and the read line here,
 * events and counts in the work cluster's LIM through one window. */
#pragma once

#include "kernel.h"
#include "ring.h"

static inline int work_pow2(uint64_t n) {
    return n != 0 && (n & (n - 1u)) == 0;
}

/* Whether the argument block's work ring is usable: none (work_cmds 0), or lines
 * after the upload region inside LIM and a remote area inside one 2 MiB window. */
static inline int work_args_valid(const volatile kmd_args_t* a) {
    if (a->work_cmds == 0) {
        return 1;
    }
    const uint64_t d = a->work_depth, e = a->work_event_depth;
    const uint64_t noc = a->work_noc, at = a->work_noc_addr;
    const uint64_t remote = KMD_WORK_EVENTS + e * sizeof(km_event_t);
    return (
        work_pow2(d) && d <= KMD_WORK_DEPTH_MAX && work_pow2(e) &&
        e <= KMD_WORK_EVENTS_MAX && (a->work_cmds & 63u) == 0 &&
        a->work_cmds >= KMD_LIM_UPLOAD + a->upload_bytes &&
        (uintptr_t)a + a->work_cmds + d * 64u + 64u <= X280_LIM_END &&
        (at & 63u) == 0 &&
        (at & (NOC_2M_WINDOW_STRIDE - 1u)) + remote <= NOC_2M_WINDOW_STRIDE &&
        (noc & 0xFFFFu) <= NOC_MAX_COORD && (noc >> 16) <= NOC_MAX_COORD
    );
}

/* Programs the window onto the work cluster's area and points source 2's rings at
 * both parts. Returns KMD_ERR_NONE (also with no ring) or KMD_ERR_WINDOW. */
static inline uint32_t work_open(kmd_t* k, uint64_t* line) {
    const volatile kmd_args_t* a = k->args;
    if (a->work_cmds == 0) {
        return KMD_ERR_NONE;
    }
    noc_tlb_2m_t programmed;
    volatile uint8_t* win = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_WORK,
            (uint32_t)(a->work_noc & 0xFFFFu),
            (uint32_t)(a->work_noc >> 16),
            a->work_noc_addr,
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (win == 0) {
        return KMD_ERR_WINDOW;
    }
    k->opened |= 1u << 3;
    if (noc_verify_tlb_2m(KMD_WIN_WORK, &programmed) != 0) {
        return KMD_ERR_WINDOW;
    }
    kmd_ring_t* r = &k->ring[KMD_SRC_WORK];
    r->depth = (uint32_t)a->work_depth;
    r->event_depth = (uint32_t)a->work_event_depth;
    r->cmds = k->lim + a->work_cmds;
    r->read = (volatile const uint32_t*)(r->cmds + r->depth * 64u);
    r->consumed = (volatile uint32_t*)(win + KMD_WORK_CONSUMED);
    r->told = (volatile uint32_t*)(win + KMD_WORK_TOLD);
    r->events = win + KMD_WORK_EVENTS;
    r->next = 1;
    r->t_record = k->ring[KMD_SRC_HOST].t_record;
    r->line = line;
    return KMD_ERR_NONE;
}

/* Finish: the last event count goes out, then the window is cleared. */
static inline void work_close(kmd_t* k) {
    if (k->opened & (1u << 3)) {
        ring_flush(&k->ring[KMD_SRC_WORK]);
        x280_fence_ow();
        noc_clear_tlb_2m(KMD_WIN_WORK);
        k->opened &= ~(1u << 3);
        k->ring[KMD_SRC_WORK].events = 0;
    }
}
