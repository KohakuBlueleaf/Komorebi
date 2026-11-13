/* Core 0's side of the work clusters' rings (sources 2 to 4): commands and read lines
 * here, events and counts in each cluster's LIM through a window of its own. */
#pragma once

#include "kernel.h"
#include "ring.h"

static inline int work_pow2(uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

/* Whether the work rings are usable: none (work_cmds 0), or all lines after the upload
 * region inside LIM and each remote area in use inside one 2 MiB window. */
static inline int work_args_valid(const volatile kmd_args_t* arguments) {
    if (arguments->work_cmds == 0) {
        return 1;
    }
    const uint64_t depth = arguments->work_depth;
    const uint64_t event_depth = arguments->work_event_depth;
    const uint64_t remote = KMD_WORK_EVENTS + event_depth * sizeof(km_event_t);
    int ok = (
        work_pow2(depth) && depth <= KMD_WORK_DEPTH_MAX && work_pow2(event_depth) &&
        event_depth <= KMD_WORK_EVENTS_MAX && (arguments->work_cmds & 63u) == 0 &&
        arguments->work_cmds >= KMD_LIM_UPLOAD + arguments->upload_bytes &&
        (uintptr_t)arguments + arguments->work_cmds +
                KMD_WORK_CLUSTERS * KMD_WORK_RING_BYTES(depth) <=
            X280_LIM_END
    );
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        const uint64_t noc = arguments->work_noc[ring_index];
        const uint64_t address = arguments->work_noc_addr[ring_index];
        ok = (
            ok &&
            (
                address == 0 ||
                (
                    (address & 63u) == 0 &&
                    (address & (NOC_2M_WINDOW_STRIDE - 1u)) + remote <=
                        NOC_2M_WINDOW_STRIDE &&
                    (noc & 0xFFFFu) <= NOC_MAX_COORD && (noc >> 16) <= NOC_MAX_COORD
                )
            )
        );
    }
    return ok;
}

/* Programs ring w's window onto its cluster's area and points source KMD_SRC_WORK + w
 * at both parts. Returns KMD_ERROR_NONE (also with no ring) or KMD_ERROR_WINDOW. */
static inline uint32_t work_open_one(
    kmd_t* dispatcher, uint32_t ring_index, uint64_t* line
) {
    const volatile kmd_args_t* arguments = dispatcher->args;
    if (arguments->work_cmds == 0 || arguments->work_noc_addr[ring_index] == 0) {
        return KMD_ERROR_NONE;
    }
    noc_tlb_2m_t programmed;
    volatile uint8_t* window = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            KMD_WIN_WORK + ring_index,
            (uint32_t)(arguments->work_noc[ring_index] & 0xFFFFu),
            (uint32_t)(arguments->work_noc[ring_index] >> 16),
            arguments->work_noc_addr[ring_index],
            NOC_ORDER_POSTED,
            &programmed
        )
    );
    if (window == 0) {
        return KMD_ERROR_WINDOW;
    }
    dispatcher->opened |= 1u << (3u + ring_index);
    if (noc_verify_tlb_2m(KMD_WIN_WORK + ring_index, &programmed) != 0) {
        return KMD_ERROR_WINDOW;
    }
    kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_WORK + ring_index];
    ring->depth = (uint32_t)arguments->work_depth;
    ring->event_depth = (uint32_t)arguments->work_event_depth;
    ring->commands = (
        dispatcher->lim + arguments->work_cmds +
        ring_index * KMD_WORK_RING_BYTES(ring->depth)
    );
    ring->read = (volatile const uint32_t*)(ring->commands + ring->depth * 64u);
    ring->consumed = (volatile uint32_t*)(window + KMD_WORK_CONSUMED);
    ring->told = (volatile uint32_t*)(window + KMD_WORK_TOLD);
    ring->events = window + KMD_WORK_EVENTS;
    ring->next = 1;
    ring->t_command = dispatcher->ring[KMD_SRC_HOST].t_command;
    ring->line = line;
    return KMD_ERROR_NONE;
}

/* Every ring in use; `lines` holds one 64-byte aligned command line per ring. */
static inline uint32_t work_open(kmd_t* dispatcher, uint64_t* lines) {
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        if (
            work_open_one(dispatcher, ring_index, lines + 8u * ring_index) !=
            KMD_ERROR_NONE
        ) {
            return KMD_ERROR_WINDOW;
        }
    }
    return KMD_ERROR_NONE;
}

/* Finish: each ring's last event count goes out, then its window is cleared. */
static inline void work_close(kmd_t* dispatcher) {
    for (uint32_t ring_index = 0; ring_index < KMD_WORK_CLUSTERS; ring_index++) {
        if (dispatcher->opened & (1u << (3u + ring_index))) {
            ring_flush(&dispatcher->ring[KMD_SRC_WORK + ring_index]);
            x280_fence_ow();
            noc_clear_tlb_2m(KMD_WIN_WORK + ring_index);
            dispatcher->opened &= ~(1u << (3u + ring_index));
            dispatcher->ring[KMD_SRC_WORK + ring_index].events = 0;
        }
    }
}
