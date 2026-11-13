/* Core 1's graphs beside the streams (included by engine.h): x280/engines/' plain-C
 * executor runs them; this part feeds it answers and writes commands and events. */
#pragma once

#include "graph_port.h"
#include "km_graph_engine.h"

/* A graph run's state on core 1. */
#define KMD_GR_FREE 0u
#define KMD_GR_RUNNING 1u
#define KMD_GR_ENDING 2u /* its end or error event waits for room in the ring */

/* In kmd_sent_t.stream: the command is graph run (stream & 1)'s; pc is its node. */
#define KMD_SENT_GRAPH 0x80u

typedef struct {
    uint32_t state; /* KMD_GR_* */
    uint32_t req, order;
    uint32_t kept;        /* a3 of its end: the graph is kept from now on */
    uint32_t wait;        /* a command was rejected: no step before the next answer */
    uint32_t round_owed;  /* a KM_EV_GRAPH event waits for room */
    uint32_t post_status; /* KMD_GR_ENDING: the event to post */
    uint32_t post_ticks;
    uint32_t
        jobs; /* queued commands it sent: its end drops those left (KMD_HANDLER_DROP) */
    uint64_t t_command; /* deadline of each command it submits, in ticks */
    kmg_state_t g;
} kmd_grun_t;

typedef struct {
    kmd_grun_t run[KM_GRAPH_MAX_ACTIVE];
} kmd_graphs_t;

_Static_assert(
    KMD_STATE_ENGINE + sizeof(kmd_engine_t) <= KMD_STATE_GRAPH,
    "core 1's engine state reaches the graphs' part"
);
_Static_assert(
    KMD_STATE_GRAPH + sizeof(kmd_graphs_t) <= KMD_STATE_RING,
    "the graph runs reach the engine's ring"
);
_Static_assert(KM_GRAPH_MAX_ACTIVE == 2u, "KMD_SENT_GRAPH keeps one bit for the run");

static inline uint8_t* graph_state_base(kmd_engine_t* e) {
    return (uint8_t*)e - KMD_STATE_ENGINE;
}

static inline kmd_graphs_t* graphs(kmd_engine_t* e) {
    return (kmd_graphs_t*)(graph_state_base(e) + KMD_STATE_GRAPH);
}

/* Core 0's table entry `i`, read only. */
static inline const volatile kmd_gentry_t* graph_entry(kmd_engine_t* e, uint32_t i) {
    const kmd_gtable_t* t = (
        (const kmd_gtable_t*)(graph_state_base(e) + KMD_STATE_GTABLE)
    );
    return (const volatile kmd_gentry_t*)&t->entry[i % KMD_GRAPH_ENTRIES];
}

/* Posts the host event {req, status, ticks} with a3 = `a3` if the ring has room. */
static int graph_post_event(
    kmd_engine_t* e, uint32_t req, uint32_t status, uint32_t ticks, uint32_t a3
) {
    if (!eng_room(e, 1)) {
        return 0;
    }
    uint64_t* l = e->line;
    l[0] = (uint64_t)KMD_HANDLER_POST << 32;
    l[1] = (uint64_t)req << 32;
    l[2] = (uint64_t)status << 32;
    l[3] = ticks | (uint64_t)a3 << 32;
    for (uint32_t j = 4; j < 8; j++) {
        l[j] = 0;
    }
    eng_write(e);
    e->posts++;
    return 1;
}

/* Ends run `r` with the event `status`, `ticks` (posted when the ring has room). */
static inline void graph_end(kmd_grun_t* r, uint32_t status, uint32_t ticks) {
    r->state = KMD_GR_ENDING;
    r->post_status = status;
    r->post_ticks = ticks;
}

/* KMD_EV_GRUN for table entry `i`: the check, the kept table, the bindings, then the
 * graph runs; a fault ends it at once with KM_EV_GRAPH_ERROR. */
static void graph_start(kmd_engine_t* e, uint32_t i) {
    kmd_graphs_t* gs = graphs(e);
    kmd_grun_t* r = 0;
    for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE && r == 0; j++) {
        r = gs->run[j].state == KMD_GR_FREE ? &gs->run[j] : 0;
    }
    const volatile kmd_gentry_t* x = graph_entry(e, i);
    x280_fence_ir();
    const uint32_t off = x->off, bytes = x->bytes, n_bind = x->n_bind;
    if (r == 0 || off > e->upload_bytes || bytes > e->upload_bytes - off) {
        return; /* core 0 admits neither */
    }
    r->req = x->req;
    /* Shared with the streams: a job's tag names one. */
    r->order = ++e->orders;
    r->wait = r->round_owed = r->jobs = 0;
    r->kept = 0;
    const void* upload = (const void*)(uintptr_t)(e->upload + off);
    uint32_t err = kmg_check(upload, bytes);
    const uint32_t keep = err ? 0u : ((const km_graph_t*)upload)->flags & KM_GRAPH_KEEP;
    if (keep && x->full) {
        err = KM_SERR_KEPT_FULL | KM_GRAPH_NO_NODE << 8;
    }
    if (err == 0 && (!x->bind_ok || n_bind > KM_GRAPH_BINDINGS)) {
        err = KM_SERR_HEADER | KM_GRAPH_NO_NODE << 8;
    }
    if (err == 0) {
        uint32_t b[KM_GRAPH_BINDINGS];
        const volatile uint32_t* w = (
            (const volatile uint32_t*)(e->upload + x->bind_off)
        );
        for (uint32_t j = 0; j < n_bind; j++) {
            b[j] = w[j];
        }
        err = kmg_start(&r->g, upload, b, n_bind, e->cores);
        err = err ? err | KM_GRAPH_NO_NODE << 8 : 0u;
    }
    if (err != 0) {
        graph_end(r, KM_EV_GRAPH_ERROR | err << 8, 0);
        return;
    }
    const uint32_t us = r->g.hdr->deadline_us;
    r->t_command = us ? (uint64_t)us * 50u : e->kernel->ring[KMD_SRC_HOST].t_command;
    r->kept = keep != 0;
    r->state = KMD_GR_RUNNING;
}

/* The submit function's context: the engine and the run. */
typedef struct {
    kmd_engine_t* e;
    kmd_grun_t* r;
    uint32_t index;
} kmd_gctx_t;

/* kmg_submit_fn: the command into the ring (after a KM_HANDLER_DEADLINE if the deadline
 * changed), or KMG_BUSY when the ring or the sent table has no room. */
static uint32_t graph_submit(void* ctx, uint32_t node, const km_command_t* rec) {
    kmd_gctx_t* c = (kmd_gctx_t*)ctx;
    kmd_engine_t* e = c->e;
    kmd_grun_t* r = c->r;
    const uint32_t lines = r->t_command != e->deadline ? 2u : 1u;
    const uint32_t cmd = e->next + lines - 1u, job = (rec->hdr >> 8) & KM_F_QUEUE;
    if (!eng_room(e, lines) || e->sent[cmd % KMD_SENT].cmd != 0) {
        return KMG_BUSY;
    }
    if (job && *e->queued + (e->jobs - *e->taken) >= KM_QUEUE_JOBS) {
        return KMG_BUSY; /* jobs in the queue and on their way to it */
    }
    if (lines == 2u) {
        const uint64_t us = r->t_command / 50u;
        eng_command(e, KM_HANDLER_DEADLINE, us == 0 ? 1u : (uint32_t)us, 0, 0);
        e->deadline = r->t_command;
    }
    /* Through a volatile pointer: a plain copy loop may become a memcpy call. */
    const volatile uint32_t* from = (const volatile uint32_t*)rec;
    uint32_t* to = (uint32_t*)e->line;
    for (uint32_t j = 0; j < sizeof(km_command_t) / 4u; j++) {
        to[j] = from[j];
    }
    if (job) {
        /* The run's tag in the slot bits, as a stream's job carries its own. */
        to[1] = (to[1] & 0xFFFFu) | (r->order & 0xFFFFu) << 16;
        e->jobs++;
        r->jobs++;
    }
    eng_write(e);
    kmd_sent_t* x = &e->sent[cmd % KMD_SENT];
    x->cmd = cmd;
    x->req = r->req;
    x->order = r->order;
    x->pc = (uint16_t)node;
    x->stream = (uint8_t)(KMD_SENT_GRAPH | c->index);
    x->result = ((rec->hdr >> 8) & KM_F_RESULT) != 0;
    e->commands++;
    return KMG_SUBMITTED;
}

/* An event from the kernel, if it is the graphs': KMD_EV_GRUN, KMD_EV_CANCEL of a
 * graph's req, or the answer to a graph's command. Returns 1 when it was. */
static int graph_event(kmd_engine_t* e, uint32_t req, uint32_t status, uint32_t ticks) {
    const uint32_t kind = status & 0xFFu, why = status >> 8;
    kmd_graphs_t* gs = graphs(e);
    if (kind == KMD_EV_GRUN) {
        graph_start(e, req);
        return 1;
    }
    if (kind == KMD_EV_CANCEL) {
        for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE; j++) {
            kmd_grun_t* r = &gs->run[j];
            if (r->state != KMD_GR_FREE && r->req == ticks) {
                kmg_stop(&r->g, KM_SERR_CANCELLED, KM_GRAPH_NO_NODE);
                return 1;
            }
        }
        return 0;
    }
    if (kind == KMD_EV_WAKE) {
        for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE; j++) {
            gs->run[j].wait = 0;
        }
        return 0; /* the streams may want it too */
    }
    kmd_sent_t* x = &e->sent[req % KMD_SENT];
    if (x->cmd != req || !(x->stream & KMD_SENT_GRAPH)) {
        return 0;
    }
    x->cmd = 0;
    kmd_grun_t* r = &gs->run[x->stream & 1u];
    const int live = r->state == KMD_GR_RUNNING && r->order == x->order;
    if (live) {
        r->wait = 0;
    }
    if (kind == KM_EV_DONE) {
        if (live) {
            kmg_complete(&r->g, x->pc, x->result ? ticks : 0u);
        }
    } else if (kind == KM_EV_TIMEOUT) {
        if (live) {
            kmg_stop(&r->g, KM_SERR_DEADLINE, x->pc);
        } else if (e->orphans < KMD_ORPHANS) {
            /* An orphan: the ordinary timeout event, with its graph's req. */
            e->orphan_req[e->orphans] = x->req;
            e->orphan_status[e->orphans] = status;
            e->orphan_ticks[e->orphans] = ticks;
            e->orphans++;
        }
    } else if (kind == KM_EV_REJECTED && why == KMD_REJECT_BUSY) {
        if (live) {
            kmg_retry(&r->g, x->pc);
            r->wait = 1;
        }
    } else if (live) {
        kmg_stop(
            &r->g, why == KMD_REJECT_GROUP ? KM_SERR_GROUP : KM_SERR_TEMPLATE, x->pc
        );
    }
    return 1;
}

/* Posts what the graphs owe: round events, ends and errors. */
static void graph_owed(kmd_engine_t* e) {
    kmd_graphs_t* gs = graphs(e);
    for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE; j++) {
        kmd_grun_t* r = &gs->run[j];
        if (r->state == KMD_GR_RUNNING && r->g.phase == KMG_ERROR) {
            /* Stopped by an event (a timeout, a cancel), also while core 1 stops. */
            const uint32_t aux = r->g.error_reason | r->g.error_node << 8;
            graph_end(r, KM_EV_GRAPH_ERROR | aux << 8, 0);
        }
        if (r->state == KMD_GR_RUNNING && r->round_owed) {
            const uint32_t done = (r->g.runs - 1u) & 0xFFFFFFu;
            r->round_owed = !graph_post_event(e, r->req, KM_EV_GRAPH | done << 8, 0, 0);
        }
        if (r->state != KMD_GR_ENDING) {
            continue;
        }
        if (r->jobs && eng_room(e, 1)) {
            eng_command(e, KMD_HANDLER_DROP, r->order & 0xFFFFu, 0, 0);
            r->jobs = 0;
        }
        if (
            !r->jobs &&
            graph_post_event(e, r->req, r->post_status, r->post_ticks, r->kept)
        ) {
            r->state = KMD_GR_FREE;
        }
    }
}

/* Whether a graph event waits to be posted. */
static int graph_owes(kmd_engine_t* e) {
    const kmd_graphs_t* gs = graphs(e);
    for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE; j++) {
        const kmd_grun_t* r = &gs->run[j];
        if (
            r->state == KMD_GR_ENDING ||
            (r->state == KMD_GR_RUNNING && (r->round_owed || r->g.phase == KMG_ERROR))
        ) {
            return 1;
        }
    }
    return 0;
}

/* The graphs' part of a pass: what they owe, then one kmg_step per running
 * graph (none while its round event or a rejected command waits). */
static void graph_step(kmd_engine_t* e) {
    kmd_graphs_t* gs = graphs(e);
    graph_owed(e);
    for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE; j++) {
        kmd_grun_t* r = &gs->run[j];
        if (
            r->state != KMD_GR_RUNNING || r->round_owed ||
            (r->wait && r->g.phase == KMG_RUNNING)
        ) {
            continue;
        }
        kmd_gctx_t c = {e, r, j};
        const uint32_t got = kmg_step(&r->g, graph_submit, &c);
        if (got == KMG_ROUND) {
            r->round_owed = 1;
        } else if (got == KMG_END) {
            const uint32_t r0 = r->g.regs[0] & 0xFFFFFFu;
            /* Every command completed: no job is left to drop. */
            r->jobs = 0;
            graph_end(r, KM_EV_GRAPH_END | r0 << 8, r->g.regs[1]);
        } else if (got == KMG_ERROR) {
            const uint32_t aux = r->g.error_reason | r->g.error_node << 8;
            graph_end(r, KM_EV_GRAPH_ERROR | aux << 8, 0);
        }
    }
    graph_owed(e);
}
