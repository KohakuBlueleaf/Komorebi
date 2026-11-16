/* Core 0's side of the stream engine on core 1: admitting KM_HANDLER_RUN, forwarding
 * RUN and KM_HANDLER_CANCEL as events on its ring, its posts, and stopping it. */
#pragma once

#include "../sources/event_ring.h"
#include "graph_port.h"
#include "../kernel.h"
#include "../result.h"
#include "../sources/ring.h"

/* Points the engine's ring at its area in the state (zeroed with the state). */
static inline void engine_open(kmd_t* dispatcher, uint64_t* line) {
    kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
    volatile uint8_t* engine_area = dispatcher->ering;
    engine_ring->commands = engine_area + KMD_ERING_CMDS;
    engine_ring->consumed = (volatile uint32_t*)(engine_area + KMD_ERING_CONSUMED);
    engine_ring->events = engine_area + KMD_ERING_EVENTS;
    engine_ring->told = (volatile uint32_t*)(engine_area + KMD_ERING_TOLD);
    engine_ring->read = (volatile const uint32_t*)(engine_area + KMD_ERING_READ);
    engine_ring->depth = KMD_ENGINE_DEPTH;
    engine_ring->event_depth = KMD_ENGINE_EVENTS;
    engine_ring->next = 1;
    engine_ring->t_command = dispatcher->ring[KMD_SRC_HOST].t_command;
    engine_ring->line = line;
}

/* Tells core 1 whether to run (`go` 1) or to return (2): the state word, then the run
 * value it waits for. */
static inline void engine_go(kmd_t* dispatcher, uint32_t go) {
    volatile uint32_t* go_words = (
        (volatile uint32_t*)(dispatcher->ering + KMD_ERING_GO)
    );
    go_words[2] = go;
    x280_fence_ow();
    *(volatile uint64_t*)go_words = dispatcher->run;
    x280_fence_ow();
}

/* Whether the engine has anything the kernel must look at: a stream, a command in
 * flight or queued, a held command. */
static inline int engine_active(const kmd_t* dispatcher) {
    const kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
    return (
        dispatcher->stream_count != 0 || engine_ring->busy_count != 0 ||
        engine_ring->queued_count != 0 || engine_ring->line_state != 0 ||
        graph_running(dispatcher)
    );
}

/* KM_HANDLER_RUN, host command `command_number`: rejected, ended with
 * KM_EVENT_STREAM_ERROR (request from the header, else 0) or forwarded. 1: wait. */
static int engine_run(
    kmd_t* dispatcher, uint32_t command_number, uint32_t offset, uint32_t bytes
) {
    const uint32_t region = dispatcher->upload_bytes;
    uint32_t request = 0;
    if (
        region >= sizeof(km_stream_t) && offset <= region - sizeof(km_stream_t) &&
        bytes >= 28u
    ) {
        /* Byte loads: `offset` may be unaligned here. */
        const volatile uint8_t* request_bytes = dispatcher->upload + offset + 24u;
        x280_fence_ir();
        request = (
            request_bytes[0] | (uint32_t)request_bytes[1] << 8 |
            (uint32_t)request_bytes[2] << 16 | (uint32_t)request_bytes[3] << 24
        );
    }
    if (dispatcher->exited || dispatcher->stream_count == KM_STREAM_MAX_ACTIVE) {
        const uint32_t why = (
            dispatcher->exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS
        );
        host_push(dispatcher, command_number, KM_EVENT_REJECTED | (why << 8), 0);
        dispatcher->rejected++;
        return 0;
    }
    uint32_t bad = (
        (offset & 63u) || offset > region || bytes > region - offset ||
        bytes < sizeof(km_stream_t) /* the engine would drop it with no verdict */
    );
    kmd_run_t* free_run = 0;
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        kmd_run_t* run = &dispatcher->runs[i];
        if (!run->active) {
            free_run = free_run ? free_run : run;
        } else if (
            (offset < run->offset + run->bytes && run->offset < offset + bytes) ||
            run->request == request
        ) {
            bad = 1;
        }
    }
    bad |= graph_claims(dispatcher, offset, bytes, request);
    if (bad) {
        host_push(
            dispatcher,
            request,
            KM_EVENT_STREAM_ERROR | (KM_STREAM_ERROR_HEADER << 8),
            0
        );
        return 0;
    }
    kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
    if (!ring_room(engine_ring)) {
        return 1;
    }
    ring_push(engine_ring, offset, KMD_EVENT_RUN, bytes);
    free_run->active = 1;
    free_run->request = request;
    free_run->offset = offset;
    free_run->bytes = bytes;
    dispatcher->stream_count++;
    dispatcher->engine_on = 1;
    dispatcher->ring[KMD_SRC_HOST].owed++;
    return 0;
}

/* KM_HANDLER_CANCEL, host command `command_number`: forwarded to the engine when a
 * stream it runs has `request`, else rejected. 0; 1: the engine's ring is full. */
static int engine_cancel(kmd_t* dispatcher, uint32_t command_number, uint32_t request) {
    for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
        if (dispatcher->runs[i].active && dispatcher->runs[i].request == request) {
            kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
            if (!ring_room(engine_ring)) {
                return 1;
            }
            ring_push(engine_ring, command_number, KMD_EVENT_CANCEL, request);
            return 0;
        }
    }
    host_push(
        dispatcher, command_number, KM_EVENT_REJECTED | (KMD_REJECT_UNKNOWN << 8), 0
    );
    dispatcher->rejected++;
    return 0;
}

/* KMD_HANDLER_POST `line`, host event {a0, a1, a2}: a stream's end uses its held room;
 * others need room, or are dropped and counted when `finishing`. Returns 1 to wait. */
static int engine_post(kmd_t* dispatcher, const uint64_t* line, int finishing) {
    const uint32_t request = (uint32_t)(line[1] >> 32);
    const uint32_t status = (uint32_t)(line[2] >> 32);
    const uint32_t ticks = (uint32_t)line[3], kind = status & 0xFFu;
    kmd_ring_t* host_ring = &dispatcher->ring[KMD_SRC_HOST];
    if (graph_post(dispatcher, line)) {
        return 0;
    }
    if (kind == KM_EVENT_STREAM_END || kind == KM_EVENT_STREAM_ERROR) {
        for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
            kmd_run_t* run = &dispatcher->runs[i];
            if (run->active && run->request == request) {
                run->active = 0;
                dispatcher->stream_count--;
                host_ring->owed--;
                host_push(dispatcher, request, status, ticks);
                return 0;
            }
        }
    }
    if (!ring_room(host_ring)) {
        if (!finishing) {
            return 1;
        }
        dispatcher->x280_cores[KMD_H_LOST]++;
        return 0;
    }
    host_push(dispatcher, request, status, ticks);
    return 0;
}

/* Finish: asks core 1 to stop and waits for it (KMD_ENGINE_STOP_TICKS), meanwhile
 * taking its posts; any other command of its ring is dropped. */
static void engine_stop(kmd_t* dispatcher) {
    kmd_ring_t* engine_ring = &dispatcher->ring[KMD_SRC_ENGINE];
    if (engine_ring->commands == 0) {
        return;
    }
    ring_flush(engine_ring);
    *(volatile uint32_t*)(dispatcher->ering + KMD_ERING_STOP) = 1u;
    x280_fence_ow();
    const uint64_t deadline = x280_mtime() + KMD_ENGINE_STOP_TICKS;
    uint32_t state = 0;
    for (int last = 0; !last;) {
        x280_fence_ir();
        state = dispatcher->x280_cores[KMD_H_ENGINE];
        last = state != KMD_ENGINE_RUNNING || x280_mtime() > deadline;
        /* At most one ring's worth per round: core 1 may still be writing. */
        for (uint32_t taken = 0; taken < KMD_ENGINE_DEPTH; taken++) {
            if (engine_ring->line_state == 0 && !ring_take(engine_ring)) {
                break;
            }
            engine_ring->line_state = 0;
            if (((uint32_t)(engine_ring->line[0] >> 32) & 0xFFu) == KMD_HANDLER_POST) {
                engine_post(dispatcher, engine_ring->line, 1);
            }
            ring_consumed(engine_ring);
        }
    }
    if (state == KMD_ENGINE_RUNNING && dispatcher->error == KMD_ERROR_NONE) {
        result_error(dispatcher, KMD_ERROR_ENGINE, 0, 0, 0, state);
    }
    if (state == KMD_ENGINE_TRAPPED && dispatcher->error == KMD_ERROR_NONE) {
        const uint32_t cause = dispatcher->x280_cores[KMD_H_TRAP_CAUSE];
        result_error(
            dispatcher,
            KMD_ERROR_TRAP,
            cause,
            dispatcher->x280_cores[KMD_H_TRAP_PC],
            1,
            0
        );
    }
}
