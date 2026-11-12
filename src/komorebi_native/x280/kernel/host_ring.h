/* Host command ring in LIM: host_depth records of 64 bytes. Command k sits in slot
 * k % host_depth; the host puts k in its seq and its three landing words. */
#pragma once

#include "kernel.h"

/* Copies command k->next into k->line if the host has stored it whole: 64-bit loads,
 * the rest of the line after a second load fence once seq shows k. Returns 1 or 0. */
static inline int hring_take(kmd_t* k) {
    const uint32_t i = k->next;
    volatile const uint64_t* w = (
        (volatile const uint64_t*)(k->lim + KMD_LIM_HRING +
        (i % k->host_depth) * sizeof(km_record_t))
    );
    uint64_t* l = k->line;
    x280_fence_ir();
    l[0] = w[0];
    if ((uint32_t)l[0] != i) {
        return 0;
    }
    x280_fence_ir();
    for (uint32_t j = 1; j < 8; j++) {
        l[j] = w[j];
    }
    /* KM_LANDED: land1..3 are the low halves of 64-bit words 2, 4 and 6. */
    return (uint32_t)l[2] == i && (uint32_t)l[4] == i && (uint32_t)l[6] == i;
}

/* The command was taken: count it and tell the host its slot is free. */
static inline void hring_consumed(kmd_t* k) {
    k->host[KMD_HOST_CONSUMED] = k->next;
    k->next++;
}
