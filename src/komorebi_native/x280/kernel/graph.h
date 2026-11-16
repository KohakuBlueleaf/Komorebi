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
    uint32_t request, order;
    uint32_t kept;        /* a3 of its end: the graph is kept from now on */
    uint32_t wait;        /* a command was rejected: no step before the next answer */
    uint32_t round_owed;  /* a KM_EVENT_GRAPH event waits for room */
    uint32_t post_status; /* KMD_GR_ENDING: the event to post */
    uint32_t post_ticks;
    uint32_t
        jobs; /* queued commands it sent: its end drops those left (KMD_HANDLER_DROP) */
    uint64_t t_command; /* deadline of each command it submits, in ticks */
    kmg_state_t executor;
} kmd_graph_run_t;

typedef struct {
    kmd_graph_run_t run[KM_GRAPH_MAX_ACTIVE];
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

static inline uint8_t* graph_state_base(kmd_engine_t* engine) {
    return (uint8_t*)engine - KMD_STATE_ENGINE;
}

static inline kmd_graphs_t* graphs(kmd_engine_t* engine) {
    return (kmd_graphs_t*)(graph_state_base(engine) + KMD_STATE_GRAPH);
}

/* Core 0's table entry `index`, read only. */
static inline const volatile kmd_gentry_t* graph_entry(
    kmd_engine_t* engine, uint32_t index
) {
    const kmd_gtable_t* table = (
        (const kmd_gtable_t*)(graph_state_base(engine) + KMD_STATE_GTABLE)
    );
    return (const volatile kmd_gentry_t*)&table->entry[index % KMD_GRAPH_ENTRIES];
}

/* Posts the host event {request, status, ticks} with a3 = `a3` if the ring has room. */
static int graph_post_event(
    kmd_engine_t* engine, uint32_t request, uint32_t status, uint32_t ticks, uint32_t a3
) {
    if (!eng_room(engine, 1)) {
        return 0;
    }
    uint64_t* line = engine->line;
    line[0] = (uint64_t)KMD_HANDLER_POST << 32;
    line[1] = (uint64_t)request << 32;
    line[2] = (uint64_t)status << 32;
    line[3] = ticks | (uint64_t)a3 << 32;
    for (uint32_t word = 4; word < 8; word++) {
        line[word] = 0;
    }
    eng_write(engine);
    engine->posts++;
    return 1;
}

/* Ends run `run` with the event `status`, `ticks` (posted when the ring has room). */
static inline void graph_end(kmd_graph_run_t* run, uint32_t status, uint32_t ticks) {
    run->state = KMD_GR_ENDING;
    run->post_status = status;
    run->post_ticks = ticks;
}

/* KMD_EVENT_GRAPH_RUN for table entry `index`: the check, the kept table, the bindings,
 * then the graph runs; a fault ends it at once with KM_EVENT_GRAPH_ERROR. */
static void graph_start(kmd_engine_t* engine, uint32_t index) {
    kmd_graphs_t* graph_runs = graphs(engine);
    kmd_graph_run_t* run = 0;
    for (uint32_t j = 0; j < KM_GRAPH_MAX_ACTIVE && run == 0; j++) {
        run = graph_runs->run[j].state == KMD_GR_FREE ? &graph_runs->run[j] : 0;
    }
    const volatile kmd_gentry_t* entry = graph_entry(engine, index);
    x280_fence_ir();
    const uint32_t offset = entry->offset;
    const uint32_t bytes = entry->bytes;
    const uint32_t binding_count = entry->binding_count;
    if (
        run == 0 || offset > engine->upload_bytes ||
        bytes > engine->upload_bytes - offset
    ) {
        return; /* core 0 admits neither */
    }
    run->request = entry->request;
    /* Shared with the streams: a job's tag names one. */
    run->order = ++engine->orders;
    run->wait = run->round_owed = run->jobs = 0;
    run->kept = 0;
    const void* upload = (const void*)(uintptr_t)(engine->upload + offset);
    uint32_t error = kmg_check(upload, bytes);
    const uint32_t keep = (
        error ? 0u : ((const km_graph_t*)upload)->flags & KM_GRAPH_KEEP
    );
    if (keep && entry->full) {
        error = KM_STREAM_ERROR_KEPT_FULL | KM_GRAPH_NO_NODE << 8;
    }
    if (error == 0 && (!entry->bind_ok || binding_count > KM_GRAPH_BINDINGS)) {
        error = KM_STREAM_ERROR_HEADER | KM_GRAPH_NO_NODE << 8;
    }
    if (error == 0) {
        uint32_t bindings[KM_GRAPH_BINDINGS];
        const volatile uint32_t*
            words = (const volatile uint32_t*)(engine->upload + entry->binding_offset);
        for (uint32_t j = 0; j < binding_count; j++) {
            bindings[j] = words[j];
        }
        error = (
            kmg_start(&run->executor, upload, bindings, binding_count, engine->cores)
        );
        kmg_island(&run->executor, engine->island_first, engine->ranks);
        error = error ? error | KM_GRAPH_NO_NODE << 8 : 0u;
    }
    if (error != 0) {
        graph_end(run, KM_EVENT_GRAPH_ERROR | error << 8, 0);
        return;
    }
    const uint32_t microseconds = run->executor.hdr->deadline_microseconds;
    run->t_command = (
        microseconds
            ? (uint64_t)microseconds * 50u
            : engine->kernel->ring[KMD_SRC_HOST].t_command
    );
    run->kept = keep != 0;
    run->state = KMD_GR_RUNNING;
}

/* The submit function's context: the engine and the run. */
typedef struct {
    kmd_engine_t* engine;
    kmd_graph_run_t* run;
    uint32_t index;
} kmd_gctx_t;

/* kmg_submit_function: the command into the ring (after a KM_HANDLER_DEADLINE if the
 * deadline changed), or KMG_BUSY when the ring or the sent table has no room. */
static uint32_t graph_submit(void* context, uint32_t node, const km_command_t* record) {
    kmd_gctx_t* submit_context = (kmd_gctx_t*)context;
    kmd_engine_t* engine = submit_context->engine;
    kmd_graph_run_t* run = submit_context->run;
    const uint32_t lines = run->t_command != engine->deadline ? 2u : 1u;
    const uint32_t command_number = engine->next + lines - 1u;
    const uint32_t job = (record->hdr >> 8) & KM_FLAG_QUEUE;
    if (
        !eng_room(engine, lines) ||
        engine->sent[command_number % KMD_SENT].command_number != 0
    ) {
        return KMG_BUSY;
    }
    if (job && *engine->queued + (engine->jobs - *engine->taken) >= KM_QUEUE_JOBS) {
        return KMG_BUSY; /* jobs in the queue and on their way to it */
    }
    if (lines == 2u) {
        const uint64_t microseconds = run->t_command / 50u;
        eng_command(
            engine,
            KM_HANDLER_DEADLINE,
            microseconds == 0 ? 1u : (uint32_t)microseconds,
            0,
            0
        );
        engine->deadline = run->t_command;
    }
    /* Through a volatile pointer: a plain copy loop may become a memcpy call. */
    const volatile uint32_t* source_words = (const volatile uint32_t*)record;
    uint32_t* destination_words = (uint32_t*)engine->line;
    for (uint32_t word = 0; word < sizeof(km_command_t) / 4u; word++) {
        destination_words[word] = source_words[word];
    }
    if (job) {
        /* The run's tag in the group's first half, as a stream's job has its own. */
        destination_words[1] &= 0xFF00FFFFu;
        destination_words
            [2] = ((destination_words[2] & 0xFFFF0000u) | (run->order & 0xFFFFu));
        engine->jobs++;
        run->jobs++;
    }
    eng_write(engine);
    kmd_sent_t* sent_command = &engine->sent[command_number % KMD_SENT];
    sent_command->command_number = command_number;
    sent_command->request = run->request;
    sent_command->order = run->order;
    sent_command->pc = (uint16_t)node;
    sent_command->stream = (uint8_t)(KMD_SENT_GRAPH | submit_context->index);
    sent_command->result = ((record->hdr >> 8) & KM_FLAG_RESULT) != 0;
    engine->commands++;
    return KMG_SUBMITTED;
}

/* An event from the kernel, if it is the graphs': KMD_EVENT_GRAPH_RUN, KMD_EVENT_CANCEL
 * of a graph's request, or the answer to a graph's command. Returns 1 when it was. */
static int graph_event(
    kmd_engine_t* engine, uint32_t request, uint32_t status, uint32_t ticks
) {
    const uint32_t kind = status & 0xFFu, why = status >> 8;
    kmd_graphs_t* graph_runs = graphs(engine);
    if (kind == KMD_EVENT_GRAPH_RUN) {
        graph_start(engine, request);
        return 1;
    }
    if (kind == KMD_EVENT_CANCEL) {
        for (uint32_t run_index = 0; run_index < KM_GRAPH_MAX_ACTIVE; run_index++) {
            kmd_graph_run_t* run = &graph_runs->run[run_index];
            if (run->state != KMD_GR_FREE && run->request == ticks) {
                kmg_stop(&run->executor, KM_STREAM_ERROR_CANCELLED, KM_GRAPH_NO_NODE);
                return 1;
            }
        }
        return 0;
    }
    if (kind == KMD_EVENT_WAKE) {
        for (uint32_t run_index = 0; run_index < KM_GRAPH_MAX_ACTIVE; run_index++) {
            graph_runs->run[run_index].wait = 0;
        }
        return 0; /* the streams may want it too */
    }
    kmd_sent_t* sent_command = &engine->sent[request % KMD_SENT];
    if (
        sent_command->command_number != request ||
        !(sent_command->stream & KMD_SENT_GRAPH)
    ) {
        return 0;
    }
    sent_command->command_number = 0;
    kmd_graph_run_t* run = &graph_runs->run[sent_command->stream & 1u];
    const int live = run->state == KMD_GR_RUNNING && run->order == sent_command->order;
    if (live) {
        run->wait = 0;
    }
    if (kind == KM_EVENT_DONE) {
        if (live) {
            kmg_complete(
                &run->executor, sent_command->pc, sent_command->result ? ticks : 0u
            );
        }
    } else if (kind == KM_EVENT_TIMEOUT) {
        if (live) {
            kmg_stop(&run->executor, KM_STREAM_ERROR_DEADLINE, sent_command->pc);
        } else if (engine->orphans < KMD_ORPHANS) {
            /* An orphan: the ordinary timeout event, with its graph's request. */
            engine->orphan_request[engine->orphans] = sent_command->request;
            engine->orphan_status[engine->orphans] = status;
            engine->orphan_ticks[engine->orphans] = ticks;
            engine->orphans++;
        }
    } else if (kind == KM_EVENT_REJECTED && why == KMD_REJECT_BUSY) {
        if (live) {
            kmg_retry(&run->executor, sent_command->pc);
            run->wait = 1;
        }
    } else if (live) {
        kmg_stop(
            &run->executor,
            why == KMD_REJECT_GROUP ? KM_STREAM_ERROR_GROUP : KM_STREAM_ERROR_TEMPLATE,
            sent_command->pc
        );
    }
    return 1;
}

/* Posts what the graphs owe: round events, ends and errors. */
static void graph_owed(kmd_engine_t* engine) {
    kmd_graphs_t* graph_runs = graphs(engine);
    for (uint32_t run_index = 0; run_index < KM_GRAPH_MAX_ACTIVE; run_index++) {
        kmd_graph_run_t* run = &graph_runs->run[run_index];
        if (run->state == KMD_GR_RUNNING && run->executor.phase == KMG_ERROR) {
            /* Stopped by an event (a timeout, a cancel), also while core 1 stops. */
            const uint32_t auxiliary = (
                run->executor.error_reason | run->executor.error_node << 8
            );
            graph_end(run, KM_EVENT_GRAPH_ERROR | auxiliary << 8, 0);
        }
        if (run->state == KMD_GR_RUNNING && run->round_owed) {
            const uint32_t done = (run->executor.runs - 1u) & 0xFFFFFFu;
            run->round_owed = (
                !graph_post_event(
                    engine, run->request, KM_EVENT_GRAPH | done << 8, 0, 0
                )
            );
        }
        if (run->state != KMD_GR_ENDING) {
            continue;
        }
        if (run->jobs && eng_room(engine, 1)) {
            eng_command(engine, KMD_HANDLER_DROP, run->order & 0xFFFFu, 0, 0);
            run->jobs = 0;
        }
        if (
            !run->jobs &&
            graph_post_event(
                engine, run->request, run->post_status, run->post_ticks, run->kept
            )
        ) {
            run->state = KMD_GR_FREE;
        }
    }
}

/* Whether a graph event waits to be posted. */
static int graph_owes(kmd_engine_t* engine) {
    const kmd_graphs_t* graph_runs = graphs(engine);
    for (uint32_t run_index = 0; run_index < KM_GRAPH_MAX_ACTIVE; run_index++) {
        const kmd_graph_run_t* run = &graph_runs->run[run_index];
        if (
            run->state == KMD_GR_ENDING ||
            (
                run->state == KMD_GR_RUNNING &&
                (run->round_owed || run->executor.phase == KMG_ERROR)
            )
        ) {
            return 1;
        }
    }
    return 0;
}

/* The graphs' part of a pass: what they owe, then one kmg_step per running
 * graph (none while its round event or a rejected command waits). */
static void graph_step(kmd_engine_t* engine) {
    kmd_graphs_t* graph_runs = graphs(engine);
    graph_owed(engine);
    for (uint32_t run_index = 0; run_index < KM_GRAPH_MAX_ACTIVE; run_index++) {
        kmd_graph_run_t* run = &graph_runs->run[run_index];
        if (
            run->state != KMD_GR_RUNNING || run->round_owed ||
            (run->wait && run->executor.phase == KMG_RUNNING)
        ) {
            continue;
        }
        kmd_gctx_t submit_context = {engine, run, run_index};
        const uint32_t got = kmg_step(&run->executor, graph_submit, &submit_context);
        if (got == KMG_ROUND) {
            run->round_owed = 1;
        } else if (got == KMG_END) {
            const uint32_t register_zero = run->executor.registers[0] & 0xFFFFFFu;
            /* Every command completed: no job is left to drop. */
            run->jobs = 0;
            graph_end(
                run, KM_EVENT_GRAPH_END | register_zero << 8, run->executor.registers[1]
            );
        } else if (got == KMG_ERROR) {
            const uint32_t auxiliary = (
                run->executor.error_reason | run->executor.error_node << 8
            );
            graph_end(run, KM_EVENT_GRAPH_ERROR | auxiliary << 8, 0);
        }
    }
    graph_owed(engine);
}
