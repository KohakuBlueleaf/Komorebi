/* Dynamic runs: two passes over the RUN team's status lines; when all are idle and
 * unchanged and sent equals consumed, the run's end record goes to the team. */

#pragma once

#include "completion.h"
#include "kernel.h"

/* After record_issue: `l` in `slot`, for command `req` of source `src`, may start the
 * dynamic run. */
KMD_HOT void dyn_issued(
    kmd_t* k, const uint64_t* l, uint32_t slot, uint32_t req, uint32_t src
) {
    kmd_dyn_t* d = &k->dyn;
    if (d->seg == 0 || ((uint32_t)(l[0] >> 32) & 0xFFu) != d->seg) {
        return;
    }
    d->active = 1;
    d->slot = slot;
    d->seq = (uint32_t)l[0];
    d->team = (uint32_t)l[1];
    d->req = req;
    d->src = src;
}

/* Whether a RUN of segment `seg` must wait: a dynamic run or its end is in flight. */
static inline int dyn_held(const kmd_t* k, uint32_t seg) {
    return k->dyn.seg != 0 && seg == k->dyn.seg && (k->dyn.active || k->dyn.end);
}

/* A status line that says gave up: its RUN record's deadline becomes now, so the
 * loop's next deadline check ends the session with KMD_ERR_DEADLINE on it. */
static inline void dyn_gave(kmd_t* k, uint32_t core) {
    k->harts[KMD_H_DYN_GAVE] = core + 1u;
    k->slot_deadline[k->dyn.slot] = 0;
    k->dyn.active = 0;
}

/* The two passes: 1 quiet, 0 not yet (a line busy, of an earlier run or changed).
 * Counter tiles are team cores like the rest: idle lines, nothing sent or consumed. */
static inline int dyn_quiet(kmd_t* k) {
    kmd_dyn_t* d = &k->dyn;
    const uint32_t f = d->team & 0xFFFFu, end = f + (d->team >> 16);
    volatile const uint32_t* lines = (volatile const uint32_t*)(k->lim + KMD_LIM_DYN);
    k->harts[KMD_H_DYN_CHECKS] = ++d->checks;
    x280_fence_ir();
    for (uint32_t c = f; c < end; c++) {
        volatile const uint32_t* l = lines + 4u * c;
        if (l[KM_DYN_LINE_RUN] != d->seq) {
            return 0;
        }
        const uint32_t state = l[KM_DYN_LINE_STATE];
        if (state == KM_DYN_GAVE_UP) {
            dyn_gave(k, c);
            return 0;
        }
        if (state != KM_DYN_IDLE) {
            return 0;
        }
        d->first[2u * c] = l[KM_DYN_LINE_SENT];
        d->first[2u * c + 1u] = l[KM_DYN_LINE_CONSUMED];
    }
    x280_fence_ir();
    uint64_t sent = 0, consumed = 0;
    for (uint32_t c = f; c < end; c++) {
        volatile const uint32_t* l = lines + 4u * c;
        if (l[KM_DYN_LINE_RUN] != d->seq || l[KM_DYN_LINE_STATE] != KM_DYN_IDLE ||
            l[KM_DYN_LINE_SENT] != d->first[2u * c] ||
            l[KM_DYN_LINE_CONSUMED] != d->first[2u * c + 1u]) {
            return 0;
        }
        sent += d->first[2u * c];
        consumed += d->first[2u * c + 1u];
    }
    return sent == consumed;
}

/* A pass while a dynamic run is in flight: once quiet and a slot is free, its end
 * record {a0: the run's seq} goes to the team under the RUN's command and source. */
static inline void dyn_turn(kmd_t* k) {
    kmd_dyn_t* d = &k->dyn;
    if (!(k->busy & (1u << d->slot)) || k->slot_seq[d->slot] != d->seq) {
        d->active = 0; /* the RUN completed or ended otherwise: nothing to end */
        return;
    }
    if (!dyn_quiet(k)) {
        return;
    }
    const uint32_t slot = slot_free(k);
    if (slot == KM_RING_SLOTS) {
        return;
    }
    const uint64_t now = x280_mtime();
    __attribute__((aligned(64))) uint64_t l[8] = {0};
    l[0] = (uint64_t)KM_SEG_DYN_END << 32;
    l[1] = d->team | (uint64_t)d->seq << 32;
    record_issue(
        k, l, slot, d->req, d->src, now, k->ring[d->src].t_record, KMD_TRACE_ISSUE
    );
    d->end = slot + 1u;
    d->active = 0;
    k->harts[KMD_H_DYN_QUIET] = (uint32_t)now;
}
