/* Core 0's side of the graph executor on core 1 (komorebi/abi/km_graph.h): the table,
 * admitting KM_HANDLER_RUN with a2 1 or 2, KM_HANDLER_CANCEL, the graphs' ends. */
#pragma once

#include "event_ring.h"
#include "kernel.h"
#include "komorebi/abi/km_graph.h"
#include "ring.h"

/* The graphs' parts of the state area: core 1's runs after its engine state, core 0's
 * table after the engine's ring. */
#define KMD_STATE_GRAPH 0x3400u
#define KMD_STATE_GTABLE 0x5A00u
/* Kernel to engine: {req table entry, ticks command}: run that entry's graph. */
#define KMD_EV_GRUN 0x18u
/* Table entries: every kept graph and every running one. */
#define KMD_GRAPH_ENTRIES (KM_GRAPH_MAX_KEPT + KM_GRAPH_MAX_ACTIVE)

/* An entry's state. */
#define KMD_GT_FREE 0u
#define KMD_GT_RUNNING 1u
#define KMD_GT_KEPT 2u /* kept and not running */

/* A graph of the table, one line. Core 0 writes it; core 1 reads it after KMD_EV_GRUN.
 */
typedef struct {
    uint32_t state; /* KMD_GT_* */
    uint32_t req, off, bytes;
    uint32_t bind_off, n_bind;
    uint32_t bind_ok; /* the binding block lies in the region */
    uint32_t keep;    /* the header's keep bit */
    uint32_t full;    /* the kept table had no room at the RUN */
    uint32_t again;   /* a run of a kept graph: it stays kept, whatever its end */
    uint32_t pad[6];
} kmd_gentry_t;

typedef struct {
    kmd_gentry_t entry[KMD_GRAPH_ENTRIES];
    uint32_t running; /* entries KMD_GT_RUNNING */
} kmd_gtable_t;

_Static_assert(sizeof(kmd_gentry_t) == 64, "a table entry is one line");
_Static_assert(
    KMD_STATE_RING + KMD_ERING_BYTES <= KMD_STATE_GTABLE &&
        KMD_STATE_GTABLE + sizeof(kmd_gtable_t) <= KMD_STATE_QUEUE,
    "the graph table lies between the engine's ring and the queue"
);

/* Core 0's graph table, in the state area beside its kmd_t. */
static inline kmd_gtable_t* gtable(const kmd_t* k) {
    return (kmd_gtable_t*)((uintptr_t)k - KMD_STATE_KERNEL + KMD_STATE_GTABLE);
}

/* Whether a graph runs: the engine's ring must be looked at. */
static inline int graph_running(const kmd_t* k) {
    return gtable(k)->running != 0;
}

/* Whether a graph of the table (running or kept) has `req` or a range that overlaps
 * [off, off + bytes): a stream may not take either. */
static int graph_claims(kmd_t* k, uint32_t off, uint32_t bytes, uint32_t req) {
    const kmd_gtable_t* t = gtable(k);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        const kmd_gentry_t* x = &t->entry[i];
        if (x->state != KMD_GT_FREE &&
            (x->req == req || (off < x->off + x->bytes && x->off < off + bytes))) {
            return 1;
        }
    }
    return 0;
}

/* Word `at` of the upload region with byte loads (it may be unaligned), 0 outside. */
static uint32_t upload_word(const kmd_t* k, uint32_t at) {
    if (k->upload_bytes < 4u || at > k->upload_bytes - 4u) {
        return 0;
    }
    const volatile uint8_t* p = k->upload + at;
    x280_fence_ir();
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static inline void graph_reject(kmd_t* k, uint32_t cmd, uint32_t why) {
    host_push(k, cmd, KM_EV_REJECTED | (why << 8), 0);
    k->rejected++;
}

/* KM_HANDLER_RUN with a2 != 0, host command `cmd` in line `l`: rejected, ended at once
 * with KM_EV_GRAPH_ERROR reason 1, or handed to core 1. Returns 1 to wait for room. */
static int graph_run(kmd_t* k, uint32_t cmd, const uint64_t* l) {
    const uint32_t a0 = (uint32_t)(l[1] >> 32), a1 = (uint32_t)(l[2] >> 32);
    const uint32_t form = (uint32_t)l[3], bind_off = (uint32_t)(l[3] >> 32);
    const uint32_t n_bind = (uint32_t)(l[4] >> 32), region = k->upload_bytes;
    kmd_gtable_t* t = gtable(k);
    if (form > KM_GRUN_KEPT) {
        graph_reject(k, cmd, KMD_REJECT_HANDLER);
        return 0;
    }
    if (k->exited || t->running == KM_GRAPH_MAX_ACTIVE) {
        graph_reject(k, cmd, k->exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS);
        return 0;
    }
    kmd_gentry_t* x = 0;
    uint32_t kept = 0, req = a0;
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        const kmd_gentry_t* y = &t->entry[i];
        kept +=
            (y->state == KMD_GT_KEPT ||
             (y->state == KMD_GT_RUNNING && (y->again || (y->keep && !y->full))));
    }
    if (form == KM_GRUN_KEPT) {
        for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES && x == 0; i++) {
            kmd_gentry_t* y = &t->entry[i];
            x = y->state == KMD_GT_KEPT && y->req == a0 ? y : 0;
        }
        if (x == 0) {
            graph_reject(k, cmd, KMD_REJECT_UNKNOWN);
            return 0;
        }
    } else {
        req = a1 >= 28u ? upload_word(k, a0 + 24u) : 0u;
        uint32_t bad = (a0 & 63u) || a0 > region || a1 > region - a0;
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            const kmd_run_t* s = &k->runs[i];
            bad |=
                (s->active &&
                 ((a0 < s->off + s->bytes && s->off < a0 + a1) || s->req == req));
        }
        bad |= graph_claims(k, a0, a1, req);
        for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES && x == 0; i++) {
            x = t->entry[i].state == KMD_GT_FREE ? &t->entry[i] : 0;
        }
        if (bad || x == 0) {
            const uint32_t why = KM_SERR_HEADER | KM_GRAPH_NO_NODE << 8;
            host_push(k, req, KM_EV_GRAPH_ERROR | why << 8, 0);
            return 0;
        }
    }
    kmd_ring_t* e = &k->ring[KMD_SRC_ENGINE];
    if (!ring_room(e)) {
        return 1;
    }
    if (form == KM_GRUN_KEPT) {
        x->again = 1;
        x->full = 0;
    } else {
        const uint32_t flags = a1 >= 48u ? upload_word(k, a0 + 44u) : 0u;
        x->req = req;
        x->off = a0;
        x->bytes = a1;
        x->keep = flags & KM_GRAPH_KEEP;
        x->full = x->keep && kept >= KM_GRAPH_MAX_KEPT;
        x->again = 0;
    }
    x->bind_off = bind_off;
    x->n_bind = n_bind;
    x->bind_ok =
        (n_bind == 0 ||
         (!(bind_off & 63u) && bind_off <= region &&
          n_bind <= (region - bind_off) / 4u));
    x->state = KMD_GT_RUNNING;
    /* Core 1 reads the entry once the event shows. */
    x280_fence_ow();
    ring_push(e, (uint32_t)(x - t->entry), KMD_EV_GRUN, cmd);
    t->running++;
    k->engine_on = 1;
    k->ring[KMD_SRC_HOST].owed++;
    return 0;
}

/* KM_HANDLER_CANCEL of `req`: core 1 cancels a running graph, a kept one not running is
 * dropped (KM_EV_GRAPH_ERROR 10). 0, 1 to wait for room, -1 if no graph has `req`. */
static int graph_cancel(kmd_t* k, uint32_t cmd, uint32_t req) {
    kmd_gtable_t* t = gtable(k);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        kmd_gentry_t* x = &t->entry[i];
        if (x->state == KMD_GT_FREE || x->req != req) {
            continue;
        }
        if (x->state == KMD_GT_KEPT) {
            x->state = KMD_GT_FREE;
            const uint32_t why = KM_SERR_CANCELLED | KM_GRAPH_NO_NODE << 8;
            host_push(k, req, KM_EV_GRAPH_ERROR | why << 8, 0);
            return 0;
        }
        kmd_ring_t* e = &k->ring[KMD_SRC_ENGINE];
        if (!ring_room(e)) {
            return 1;
        }
        ring_push(e, cmd, KMD_EV_CANCEL, req);
        return 0;
    }
    return -1;
}

/* A graph's end or error from core 1 (KMD_HANDLER_POST `l`, a3 = 1 if now kept): uses
 * the event room held for it, frees or keeps its entry. Returns 0 if not a graph's. */
static int graph_post(kmd_t* k, const uint64_t* l) {
    const uint32_t req = (uint32_t)(l[1] >> 32), status = (uint32_t)(l[2] >> 32);
    const uint32_t kind = status & 0xFFu;
    if (kind != KM_EV_GRAPH_END && kind != KM_EV_GRAPH_ERROR) {
        return 0;
    }
    kmd_gtable_t* t = gtable(k);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        kmd_gentry_t* x = &t->entry[i];
        if (x->state == KMD_GT_RUNNING && x->req == req) {
            const uint32_t kept = x->again || (uint32_t)(l[3] >> 32) != 0;
            x->state = kept ? KMD_GT_KEPT : KMD_GT_FREE;
            t->running--;
            k->ring[KMD_SRC_HOST].owed--;
            host_push(k, req, status, (uint32_t)l[3]);
            return 1;
        }
    }
    return 0;
}
