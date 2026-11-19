/* Core 0's side of the graph executor on core 1 (komorebi/abi/km_graph.h): the table,
 * admitting KM_HANDLER_RUN with a2 1 or 2, KM_HANDLER_CANCEL, the graphs' ends. */
#pragma once

#include "../sources/event_ring.h"
#include "../kernel.h"
#include "komorebi/abi/km_graph.h"
#include "../sources/ring.h"

/* The graphs' parts of the state area: core 1's runs after its engine state, core 0's
 * table after the engine's ring. */
#define KMD_STATE_GRAPH 0x3640u
#define KMD_STATE_GTABLE 0x5A00u
/* Kernel to engine: {request table entry, ticks command}: run that entry's graph. */
#define KMD_EVENT_GRAPH_RUN 0x18u
/* Table entries: every kept graph and every running one. */
#define KMD_GRAPH_ENTRIES (KM_GRAPH_MAX_KEPT + KM_GRAPH_MAX_ACTIVE)

/* An entry's state. */
#define KMD_GT_FREE 0u
#define KMD_GT_RUNNING 1u
#define KMD_GT_KEPT 2u /* kept and not running */

/* A graph of the table, one line. Core 0 writes it; core 1 reads it after
 * KMD_EVENT_GRAPH_RUN. */
typedef struct {
    uint32_t state; /* KMD_GT_* */
    uint32_t request, offset, bytes;
    uint32_t binding_offset, binding_count;
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
static inline kmd_gtable_t* gtable(const kmd_t* dispatcher) {
    return (kmd_gtable_t*)((uintptr_t)dispatcher - KMD_STATE_KERNEL + KMD_STATE_GTABLE);
}

/* Whether a graph runs: the engine's ring must be looked at. */
KMD_COLD int graph_running(const kmd_t* dispatcher) {
    return gtable(dispatcher)->running != 0;
}

/* Whether a graph of the table (running or kept) has `request` or a range that overlaps
 * [offset, offset + bytes): a stream may not take either. */
static int graph_claims(
    kmd_t* dispatcher, uint32_t offset, uint32_t bytes, uint32_t request
) {
    const kmd_gtable_t* table = gtable(dispatcher);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        const kmd_gentry_t* entry = &table->entry[i];
        if (
            entry->state != KMD_GT_FREE &&
            (
                entry->request == request ||
                (
                    offset < entry->offset + entry->bytes &&
                    entry->offset < offset + bytes
                )
            )
        ) {
            return 1;
        }
    }
    return 0;
}

/* Word `offset` of the upload region with byte loads (it may be unaligned), 0 outside.
 */
static uint32_t upload_word(const kmd_t* dispatcher, uint32_t offset) {
    if (dispatcher->upload_bytes < 4u || offset > dispatcher->upload_bytes - 4u) {
        return 0;
    }
    const volatile uint8_t* bytes = dispatcher->upload + offset;
    x280_fence_ir();
    return (
        bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 |
        (uint32_t)bytes[3] << 24
    );
}

static inline void graph_reject(
    kmd_t* dispatcher, uint32_t command_number, uint32_t why
) {
    host_push(dispatcher, command_number, KM_EVENT_REJECTED | (why << 8), 0);
    dispatcher->rejected++;
}

/* KM_HANDLER_RUN with a2 != 0, host command `command_number` in line `line`: rejected,
 * ended with KM_EVENT_GRAPH_ERROR reason 1, or handed to core 1. 1: wait for room. */
KMD_COLD int graph_run(
    kmd_t* dispatcher, uint32_t command_number, const uint64_t* line
) {
    const uint32_t request_or_offset = (uint32_t)(line[1] >> 32);
    const uint32_t bytes = (uint32_t)(line[2] >> 32);
    const uint32_t form = (uint32_t)line[3], binding_offset = (uint32_t)(line[3] >> 32);
    const uint32_t binding_count = (uint32_t)(line[4] >> 32);
    const uint32_t region = dispatcher->upload_bytes;
    kmd_gtable_t* table = gtable(dispatcher);
    if (form > KM_GRAPH_RUN_KEPT) {
        graph_reject(dispatcher, command_number, KMD_REJECT_HANDLER);
        return 0;
    }
    if (dispatcher->exited || table->running == KM_GRAPH_MAX_ACTIVE) {
        graph_reject(
            dispatcher,
            command_number,
            dispatcher->exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS
        );
        return 0;
    }
    kmd_gentry_t* entry = 0;
    uint32_t kept = 0, request = request_or_offset;
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        const kmd_gentry_t* candidate = &table->entry[i];
        kept += (
            candidate->state == KMD_GT_KEPT ||
            (
                candidate->state == KMD_GT_RUNNING &&
                (candidate->again || (candidate->keep && !candidate->full))
            )
        );
    }
    if (form == KM_GRAPH_RUN_KEPT) {
        for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES && entry == 0; i++) {
            kmd_gentry_t* candidate = &table->entry[i];
            entry = (
                candidate->state == KMD_GT_KEPT &&
                    candidate->request == request_or_offset
                    ? candidate
                    : 0
            );
        }
        if (entry == 0) {
            graph_reject(dispatcher, command_number, KMD_REJECT_UNKNOWN);
            return 0;
        }
    } else {
        request = bytes >= 28u ? upload_word(dispatcher, request_or_offset + 24u) : 0u;
        uint32_t bad = (
            (request_or_offset & 63u) || request_or_offset > region ||
            bytes > region - request_or_offset
        );
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            const kmd_run_t* run = &dispatcher->runs[i];
            bad |= (
                run->active &&
                (
                    (
                        request_or_offset < run->offset + run->bytes &&
                        run->offset < request_or_offset + bytes
                    ) ||
                    run->request == request
                )
            );
        }
        bad |= graph_claims(dispatcher, request_or_offset, bytes, request);
        for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES && entry == 0; i++) {
            entry = table->entry[i].state == KMD_GT_FREE ? &table->entry[i] : 0;
        }
        if (bad || entry == 0) {
            const uint32_t why = KM_STREAM_ERROR_HEADER | KM_GRAPH_NO_NODE << 8;
            host_push(dispatcher, request, KM_EVENT_GRAPH_ERROR | why << 8, 0);
            return 0;
        }
    }
    kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
    if (!ring_room(engine_ring)) {
        return 1;
    }
    if (form == KM_GRAPH_RUN_KEPT) {
        entry->again = 1;
        entry->full = 0;
    } else {
        const uint32_t flags = (
            bytes >= 48u ? upload_word(dispatcher, request_or_offset + 44u) : 0u
        );
        entry->request = request;
        entry->offset = request_or_offset;
        entry->bytes = bytes;
        entry->keep = flags & KM_GRAPH_KEEP;
        entry->full = entry->keep && kept >= KM_GRAPH_MAX_KEPT;
        entry->again = 0;
    }
    entry->binding_offset = binding_offset;
    entry->binding_count = binding_count;
    entry->bind_ok = (
        binding_count == 0 ||
        (
            !(binding_offset & 63u) && binding_offset <= region &&
            binding_count <= (region - binding_offset) / 4u
        )
    );
    entry->state = KMD_GT_RUNNING;
    /* Core 1 reads the entry once the event shows. */
    x280_fence_ow();
    ring_push(
        engine_ring,
        (uint32_t)(entry - table->entry),
        KMD_EVENT_GRAPH_RUN,
        command_number
    );
    table->running++;
    dispatcher->engine_on = 1;
    dispatcher->ring[KMD_SRC_HOST].owed++;
    return 0;
}

/* KM_HANDLER_CANCEL of `request`: core 1 cancels a running graph, a kept one is dropped
 * (KM_EVENT_GRAPH_ERROR 10). 0, 1: wait for room, -1: no graph has `request`. */
KMD_COLD int graph_cancel(
    kmd_t* dispatcher, uint32_t command_number, uint32_t request
) {
    kmd_gtable_t* table = gtable(dispatcher);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        kmd_gentry_t* entry = &table->entry[i];
        if (entry->state == KMD_GT_FREE || entry->request != request) {
            continue;
        }
        if (entry->state == KMD_GT_KEPT) {
            entry->state = KMD_GT_FREE;
            const uint32_t why = KM_STREAM_ERROR_CANCELLED | KM_GRAPH_NO_NODE << 8;
            host_push(dispatcher, request, KM_EVENT_GRAPH_ERROR | why << 8, 0);
            return 0;
        }
        kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
        if (!ring_room(engine_ring)) {
            return 1;
        }
        ring_push(engine_ring, command_number, KMD_EVENT_CANCEL, request);
        return 0;
    }
    return -1;
}

/* A graph's end or error from core 1 (KMD_HANDLER_POST `line`, a3 = 1 if now kept):
 * uses its held event room, frees or keeps its entry. 0: not a graph's. */
static int graph_post(kmd_t* dispatcher, const uint64_t* line) {
    const uint32_t request = (uint32_t)(line[1] >> 32);
    const uint32_t status = (uint32_t)(line[2] >> 32);
    const uint32_t kind = status & 0xFFu;
    if (kind != KM_EVENT_GRAPH_END && kind != KM_EVENT_GRAPH_ERROR) {
        return 0;
    }
    kmd_gtable_t* table = gtable(dispatcher);
    for (uint32_t i = 0; i < KMD_GRAPH_ENTRIES; i++) {
        kmd_gentry_t* entry = &table->entry[i];
        if (entry->state == KMD_GT_RUNNING && entry->request == request) {
            const uint32_t kept = entry->again || (uint32_t)(line[3] >> 32) != 0;
            entry->state = kept ? KMD_GT_KEPT : KMD_GT_FREE;
            table->running--;
            dispatcher->ring[KMD_SRC_HOST].owed--;
            host_push(dispatcher, request, status, (uint32_t)line[3]);
            return 1;
        }
    }
    return 0;
}
