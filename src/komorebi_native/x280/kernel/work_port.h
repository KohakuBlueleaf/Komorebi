/* Core 0's side of the work clusters' rings (sources 2 to 4): commands and read lines
 * here, events and counts in each cluster's LIM through a window of its own. */
#pragma once

#include "kernel.h"
#include "ring.h"

static inline int work_pow2(uint64_t n) {
    return n != 0 && (n & (n - 1u)) == 0;
}

/* Whether the work rings are usable: none (work_cmds 0), or all lines after the upload
 * region inside LIM and each remote area in use inside one 2 MiB window. */
static inline int work_args_valid(const volatile kmd_args_t* a) {
    if (a->work_cmds == 0) {
        return 1;
    }
    const uint64_t d = a->work_depth, e = a->work_event_depth;
    const uint64_t remote = KMD_WORK_EVENTS + e * sizeof(km_event_t);
    int ok = (
        work_pow2(d) && d <= KMD_WORK_DEPTH_MAX && work_pow2(e) &&
        e <= KMD_WORK_EVENTS_MAX && (a->work_cmds & 63u) == 0 &&
        a->work_cmds >= KMD_LIM_UPLOAD + a->upload_bytes &&
        (uintptr_t)a + a->work_cmds + KMD_WORK_CLUSTERS * KMD_WORK_RING_BYTES(d) <=
            X280_LIM_END
    );
    for (uint32_t w = 0; w < KMD_WORK_CLUSTERS; w++) {
        const uint64_t noc = a->work_noc[w], at = a->work_noc_addr[w];
        ok = (
            ok &&
            (
                at == 0 ||
                (
                    (at & 63u) == 0 &&
                    (at & (NOC_2M_WINDOW_STRIDE - 1u)) + remote <=
                        NOC_2M_WINDOW_STRIDE &&
                    (noc & 0xFFFFu) <= NOC_MAX_COORD && (noc >> 16) <= NOC_MAX_COORD
                )
            )
        );
    }
    return ok;
}

/* Programs ring w's window onto its cluster's area and points source KMD_SRC_WORK + w
 * at both parts. Returns KMD_ERR_NONE (also with no ring) or KMD_ERR_WINDOW. */
static inline uint32_t work_open_one(kmd_t* k, uint32_t w, uint64_t* line) {
    const volatile kmd_args_t* a = k->args;
    if (a->work_cmds == 0 || a->work_noc_addr[w] == 0) {
        return KMD_ERR_NONE;
    }
    noc_tlb_2m_t programmed;
    volatile uint8_t* win = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_WORK + w,
            (uint32_t)(a->work_noc[w] & 0xFFFFu),
            (uint32_t)(a->work_noc[w] >> 16),
            a->work_noc_addr[w],
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (win == 0) {
        return KMD_ERR_WINDOW;
    }
    k->opened |= 1u << (3u + w);
    if (noc_verify_tlb_2m(KMD_WIN_WORK + w, &programmed) != 0) {
        return KMD_ERR_WINDOW;
    }
    kmd_ring_t* r = &k->ring[KMD_SRC_WORK + w];
    r->depth = (uint32_t)a->work_depth;
    r->event_depth = (uint32_t)a->work_event_depth;
    r->cmds = k->lim + a->work_cmds + w * KMD_WORK_RING_BYTES(r->depth);
    r->read = (volatile const uint32_t*)(r->cmds + r->depth * 64u);
    r->consumed = (volatile uint32_t*)(win + KMD_WORK_CONSUMED);
    r->told = (volatile uint32_t*)(win + KMD_WORK_TOLD);
    r->events = win + KMD_WORK_EVENTS;
    r->next = 1;
    r->t_command = k->ring[KMD_SRC_HOST].t_command;
    r->line = line;
    return KMD_ERR_NONE;
}

/* Every ring in use; `lines` holds one 64-byte aligned command line per ring. */
static inline uint32_t work_open(kmd_t* k, uint64_t* lines) {
    for (uint32_t w = 0; w < KMD_WORK_CLUSTERS; w++) {
        if (work_open_one(k, w, lines + 8u * w) != KMD_ERR_NONE) {
            return KMD_ERR_WINDOW;
        }
    }
    return KMD_ERR_NONE;
}

/* Finish: each ring's last event count goes out, then its window is cleared. */
static inline void work_close(kmd_t* k) {
    for (uint32_t w = 0; w < KMD_WORK_CLUSTERS; w++) {
        if (k->opened & (1u << (3u + w))) {
            ring_flush(&k->ring[KMD_SRC_WORK + w]);
            x280_fence_ow();
            noc_clear_tlb_2m(KMD_WIN_WORK + w);
            k->opened &= ~(1u << (3u + w));
            k->ring[KMD_SRC_WORK + w].events = 0;
        }
    }
}
