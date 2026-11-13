/* Core 0's side of the stream engine on core 1: admitting KM_HANDLER_RUN, forwarding
 * RUN and KM_HANDLER_CANCEL as events on its ring, its posts, and stopping it. */
#pragma once

#include "event_ring.h"
#include "graph_port.h"
#include "kernel.h"
#include "result.h"
#include "ring.h"

/* Points the engine's ring at its area in the state (zeroed with the state). */
static inline void engine_open(kmd_t* k, uint64_t* line) {
    kmd_ring_t* r = &k->ring[KMD_SRC_ENGINE];
    volatile uint8_t* e = k->ering;
    r->cmds = e + KMD_ERING_CMDS;
    r->consumed = (volatile uint32_t*)(e + KMD_ERING_CONSUMED);
    r->events = e + KMD_ERING_EVENTS;
    r->told = (volatile uint32_t*)(e + KMD_ERING_TOLD);
    r->read = (volatile const uint32_t*)(e + KMD_ERING_READ);
    r->depth = KMD_ENGINE_DEPTH;
    r->event_depth = KMD_ENGINE_EVENTS;
    r->next = 1;
    r->t_command = k->ring[KMD_SRC_HOST].t_command;
    r->line = line;
}

/* Tells core 1 whether to run (`go` 1) or to return (2): the state word, then the run
 * value it waits for. */
static inline void engine_go(kmd_t* k, uint32_t go) {
    volatile uint32_t* g = (volatile uint32_t*)(k->ering + KMD_ERING_GO);
    g[2] = go;
    x280_fence_ow();
    *(volatile uint64_t*)g = k->run;
    x280_fence_ow();
}

/* Whether the engine has anything the kernel must look at: a stream, a command in
 * flight or queued, a held command. */
static inline int engine_active(const kmd_t* k) {
    const kmd_ring_t* r = &k->ring[KMD_SRC_ENGINE];
    return (
        k->nstreams != 0 || r->nbusy != 0 || r->nqueued != 0 || r->line_state != 0 ||
        graph_running(k)
    );
}

/* KM_HANDLER_RUN, host command `cmd`: rejected, ended with KM_EVENT_STREAM_ERROR (req
 * from the header if in the region, else 0) or forwarded. 1: wait for room. */
static int engine_run(kmd_t* k, uint32_t cmd, uint32_t off, uint32_t bytes) {
    const uint32_t region = k->upload_bytes;
    uint32_t req = 0;
    if (
        region >= sizeof(km_stream_t) && off <= region - sizeof(km_stream_t) &&
        bytes >= 28u
    ) {
        /* Byte loads: `off` may be unaligned here. */
        const volatile uint8_t* p = k->upload + off + 24u;
        x280_fence_ir();
        req = p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    }
    if (k->exited || k->nstreams == KM_STREAM_MAX_ACTIVE) {
        const uint32_t why = k->exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS;
        host_push(k, cmd, KM_EVENT_REJECTED | (why << 8), 0);
        k->rejected++;
        return 0;
    }
    uint32_t bad = (off & 63u) || off > region || bytes > region - off;
    kmd_run_t* free_run = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_run_t* s = &k->runs[i];
        if (!s->active) {
            free_run = free_run ? free_run : s;
        } else if ((off < s->off + s->bytes && s->off < off + bytes) || s->req == req) {
            bad = 1;
        }
    }
    bad |= graph_claims(k, off, bytes, req);
    if (bad) {
        host_push(k, req, KM_EVENT_STREAM_ERROR | (KM_STREAM_ERROR_HEADER << 8), 0);
        return 0;
    }
    kmd_ring_t* e = &k->ring[KMD_SRC_ENGINE];
    if (!ring_room(e)) {
        return 1;
    }
    ring_push(e, off, KMD_EVENT_RUN, bytes);
    free_run->active = 1;
    free_run->req = req;
    free_run->off = off;
    free_run->bytes = bytes;
    k->nstreams++;
    k->engine_on = 1;
    k->ring[KMD_SRC_HOST].owed++;
    return 0;
}

/* KM_HANDLER_CANCEL, host command `cmd`: forwarded to the engine when a stream it runs
 * has `req`, else rejected. Returns 0, or 1 when the engine's ring has no room. */
static int engine_cancel(kmd_t* k, uint32_t cmd, uint32_t req) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        if (k->runs[i].active && k->runs[i].req == req) {
            kmd_ring_t* e = &k->ring[KMD_SRC_ENGINE];
            if (!ring_room(e)) {
                return 1;
            }
            ring_push(e, cmd, KMD_EVENT_CANCEL, req);
            return 0;
        }
    }
    host_push(k, cmd, KM_EVENT_REJECTED | (KMD_REJECT_UNKNOWN << 8), 0);
    k->rejected++;
    return 0;
}

/* KMD_HANDLER_POST `l`, host event {a0, a1, a2}: a stream's end uses its held room;
 * others need room, or are dropped and counted when `finishing`. Returns 1 to wait. */
static int engine_post(kmd_t* k, const uint64_t* l, int finishing) {
    const uint32_t req = (uint32_t)(l[1] >> 32), status = (uint32_t)(l[2] >> 32);
    const uint32_t ticks = (uint32_t)l[3], kind = status & 0xFFu;
    kmd_ring_t* h = &k->ring[KMD_SRC_HOST];
    if (graph_post(k, l)) {
        return 0;
    }
    if (kind == KM_EVENT_STREAM_END || kind == KM_EVENT_STREAM_ERROR) {
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            kmd_run_t* s = &k->runs[i];
            if (s->active && s->req == req) {
                s->active = 0;
                k->nstreams--;
                h->owed--;
                host_push(k, req, status, ticks);
                return 0;
            }
        }
    }
    if (!ring_room(h)) {
        if (!finishing) {
            return 1;
        }
        k->x280_cores[KMD_H_LOST]++;
        return 0;
    }
    host_push(k, req, status, ticks);
    return 0;
}

/* Finish: asks core 1 to stop and waits for it (KMD_ENGINE_STOP_TICKS), meanwhile
 * taking its posts; any other command of its ring is dropped. */
static void engine_stop(kmd_t* k) {
    kmd_ring_t* r = &k->ring[KMD_SRC_ENGINE];
    if (r->cmds == 0) {
        return;
    }
    ring_flush(r);
    *(volatile uint32_t*)(k->ering + KMD_ERING_STOP) = 1u;
    x280_fence_ow();
    const uint64_t by = x280_mtime() + KMD_ENGINE_STOP_TICKS;
    uint32_t state = 0;
    for (int last = 0; !last;) {
        x280_fence_ir();
        state = k->x280_cores[KMD_H_ENGINE];
        last = state != KMD_ENGINE_RUNNING || x280_mtime() > by;
        /* At most one ring's worth per round: core 1 may still be writing. */
        for (uint32_t n = 0; n < KMD_ENGINE_DEPTH; n++) {
            if (r->line_state == 0 && !ring_take(r)) {
                break;
            }
            r->line_state = 0;
            if (((uint32_t)(r->line[0] >> 32) & 0xFFu) == KMD_HANDLER_POST) {
                engine_post(k, r->line, 1);
            }
            ring_consumed(r);
        }
    }
    if (state == KMD_ENGINE_RUNNING && k->error == KMD_ERROR_NONE) {
        result_error(k, KMD_ERROR_ENGINE, 0, 0, 0, state);
    }
    if (state == KMD_ENGINE_TRAPPED && k->error == KMD_ERROR_NONE) {
        const uint32_t cause = k->x280_cores[KMD_H_TRAP_CAUSE];
        result_error(k, KMD_ERROR_TRAP, cause, k->x280_cores[KMD_H_TRAP_PC], 1, 0);
    }
}
