/* Dispatcher, X280 cluster 0: core 0 boots, loops over acks and the sources'
 * commands, finishes; core 1 runs the stream engine; cores 2 and 3 return. */
#include <stdint.h>

#include "completion.h"
#include "dynamic.h"
#include "engine.h"
#include "engine_port.h"
#include "event_ring.h"
#include "graph_port.h"
#include "grid_port.h"
#include "inspect.h"
#include "kernel.h"
#include "link.h"
#include "queue.h"
#include "reset.h"
#include "result.h"
#include "ring.h"
#include "trap.h"
#include "watch.h"
#include "work_port.h"

/* Waits until `word` holds `value`, the runtime asks for a stop, or `deadline`. */
static kmd_wait_t wait_word(
    const kmd_t* dispatcher,
    volatile const uint32_t* word,
    uint32_t value,
    uint64_t deadline
) {
    uint32_t spins = 0;
    for (;;) {
        x280_fence_ir();
        if (*word == value) {
            return KMD_OK;
        }
        const kmd_wait_t outcome = kmd_poll(dispatcher, &spins, deadline);
        if (outcome != KMD_OK) {
            return outcome;
        }
    }
}

static int pow2(uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

/* Dynamic handlers, one per byte: each 0 or a program handler; above them none or the
 * X280 ranks (KMD_X280_RANKS), on a cluster with a work ring. */
static int dynamic_valid(const volatile kmd_args_t* arguments) {
    const uint64_t handlers = arguments->messages_handler;
    for (uint32_t index = 0; index < KM_DYNAMIC_MAX; index++) {
        if ((uint32_t)(handlers >> 8u * index & 0xFFu) > KM_HANDLER_USER_MAX) {
            return 0;
        }
    }
    const uint32_t cluster = KMD_X280_RANKS_CLUSTER(handlers);
    const uint32_t count = KMD_X280_RANKS_COUNT(handlers);
    if (handlers >> 40 != 0 || (cluster == 0) != (count == 0)) {
        return 0;
    }
    return (
        count == 0 ||
        (
            cluster <= KMD_WORK_CLUSTERS && count <= KM_X280_RANKS_MAX &&
            arguments->cores <= KM_X280_RANK_CORE(0) && arguments->work_cmds != 0 &&
            arguments->work_noc_addr[cluster - 1] != 0
        )
    );
}

/* Checks the argument block against this run: ring depths are powers of two, so the
 * slot numbering survives the wrap of the 32-bit counters. */
static int args_valid(const volatile kmd_args_t* arguments, uint64_t run) {
    if (
        x280_hart_args_bytes() < sizeof(kmd_args_t) || arguments->run_value != run ||
        arguments->version != KMD_ARGS_VERSION
    ) {
        return 0;
    }
    const uint64_t host_bytes = (
        KMD_HOST_RING + arguments->event_depth * sizeof(km_event_t)
    );
    return (
        arguments->depth >= 1 && arguments->depth <= KM_RING_SLOTS &&
        pow2(arguments->host_depth) && arguments->host_depth <= KMD_HOST_DEPTH_MAX &&
        pow2(arguments->event_depth) && arguments->columns >= 1 &&
        arguments->columns <= KMD_READY_MAX && arguments->cores >= 1 &&
        arguments->cores <= KM_CORES_MAX && arguments->panels >= 1 &&
        arguments->panels <= 2 && arguments->t_start != 0 &&
        arguments->t_command != 0 && arguments->state_ddr != 0 &&
        (arguments->upload_bytes & 63u) == 0 &&
        arguments->upload_bytes <= X280_LIM_SIZE &&
        (uintptr_t)arguments + KMD_LIM_UPLOAD + arguments->upload_bytes <=
            X280_LIM_END &&
        (arguments->state_ddr & 63u) == 0 && (arguments->host_noc_addr & 63u) == 0 &&
        (arguments->host_noc_addr & (NOC_2M_WINDOW_STRIDE - 1u)) + host_bytes <=
            NOC_2M_WINDOW_STRIDE &&
        work_args_valid(arguments) && dynamic_valid(arguments) &&
        arguments->programs <= 1u &&
        (
            arguments->island == 0 ||
            (
                (arguments->island >> 48) == 0u &&
                ((arguments->island >> 16) & 0xFFFFu) >= 1u &&
                ((arguments->island >> 32) & 0xFFFFu) +
                        ((arguments->island >> 16) & 0xFFFFu) <=
                    arguments->cores &&
                (arguments->island & 0xFFFFu) + ((arguments->island >> 16) & 0xFFFFu) <=
                    0xFFFFu
            )
        )
    );
}

/* Start-up: host window, hello, panels (so a failed READY wait can stop the grid),
 * READY. Returns KMD_ERROR_NONE or an error. */
static uint32_t boot(kmd_t* dispatcher) {
    const volatile kmd_args_t* arguments = dispatcher->args;
    if (events_open(dispatcher) != KMD_ERROR_NONE) {
        result_error(dispatcher, KMD_ERROR_WINDOW, 0, 0, 0, 0);
        return dispatcher->error;
    }
    events_word(dispatcher, KMD_HOST_STATE, KMD_STATE_BOOT);
    const uint64_t deadline = dispatcher->start_mtime + arguments->t_start;
    volatile const uint32_t* mailbox = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_MAILBOX)
    );
    result_where(dispatcher, KMD_WHERE_HELLO);
    kmd_wait_t outcome = wait_word(dispatcher, mailbox, 1u, deadline);
    if (outcome != KMD_OK) {
        result_error(
            dispatcher,
            outcome == KMD_STOP ? KMD_ERROR_STOPPED : KMD_ERROR_START,
            0,
            0,
            0,
            0
        );
        return dispatcher->error;
    }
    x280_fence_ir();
    const uint32_t control = mailbox[1];
    result_word(dispatcher, KMD_R_CONTROL, control);
    if (
        (control & 63u) || control < KMD_CONTROL_FLOOR ||
        control > KMD_L1_BYTES - KM_CONTROL_BYTES
    ) {
        result_error(dispatcher, KMD_ERROR_ALIGN, 0, 0, 0, control);
        return dispatcher->error;
    }
    dispatcher->control_address = control;
    if (grid_open(dispatcher, control) != KMD_ERROR_NONE) {
        result_error(dispatcher, KMD_ERROR_WINDOW, 0, 0, 0, 0);
        return dispatcher->error;
    }
    result_where(dispatcher, KMD_WHERE_READY);
    for (uint32_t column = 0; column < (uint32_t)arguments->columns; column++) {
        outcome = (
            wait_word(
                dispatcher,
                (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_READY + column * 16u),
                KM_READY,
                deadline
            )
        );
        if (outcome != KMD_OK) {
            const uint32_t error = (
                outcome == KMD_STOP ? KMD_ERROR_STOPPED : KMD_ERROR_START
            );
            result_error(dispatcher, error, 0, 0, 0, column);
            return dispatcher->error;
        }
    }
    return KMD_ERROR_NONE;
}

#ifdef KMD_VERIFY_LINES
/* Test build: checks each word of the user command in the host's line against its
 * command number. */
static void verify_line(kmd_t* dispatcher) {
    const kmd_ring_t* ring = &dispatcher->ring[KMD_SRC_HOST];
    const uint64_t* line = ring->line;
    const uint32_t number = ring->next, header = (uint32_t)(line[0] >> 32);
    if ((header & 0xFFu) > KM_HANDLER_USER_MAX) {
        return;
    }
    const uint32_t words[12] = {
        header & 0xFFFFu,
        (uint32_t)line[1],
        (uint32_t)(line[1] >> 32),
        (uint32_t)(line[2] >> 32),
        (uint32_t)line[3],
        (uint32_t)(line[3] >> 32),
        (uint32_t)(line[4] >> 32),
        (uint32_t)line[5],
        (uint32_t)(line[5] >> 32),
        (uint32_t)(line[6] >> 32),
        (uint32_t)line[7],
        (uint32_t)(line[7] >> 32)
    };
    uint32_t bad = (
        words[0] != 0 || words[1] != kmd_verify_group(number, dispatcher->cores)
    );
    for (uint32_t argument = 0; argument < 10; argument++) {
        bad |= words[2 + argument] != kmd_verify_word(number, argument);
    }
    dispatcher->checked++;
    if (bad && dispatcher->bad_lines++ == 0) {
        dispatcher->bad_first = number;
    }
}

/* Test build: KMD_HANDLER_CHECK_UPLOAD. Answers PONG with auxiliary = words that differ
 * (at most 0xFFFFFF) and ticks = the first one's index from `offset` (~0: none). */
static void upload_check(
    kmd_t* dispatcher, uint32_t request, uint32_t offset, uint32_t bytes, uint32_t seed
) {
    if (
        ((offset | bytes) & 63u) || offset > dispatcher->upload_bytes ||
        bytes > dispatcher->upload_bytes - offset
    ) {
        host_push(dispatcher, request, KM_EVENT_REJECTED | (KMD_REJECT_REGION << 8), 0);
        dispatcher->rejected++;
        return;
    }
    const volatile uint64_t* words = (
        (const volatile uint64_t*)(dispatcher->upload + offset)
    );
    uint32_t bad = 0, first = ~0u;
    x280_fence_ir();
    for (uint32_t i = 0; i < bytes / 8u; i++) {
        const uint64_t word = words[i];
        const uint32_t word_index = offset / 4u + 2u * i;
        const uint32_t low_differs = (
            (uint32_t)word != kmd_verify_word(seed, word_index)
        );
        const uint32_t high_differs = (
            (uint32_t)(word >> 32) != kmd_verify_word(seed, word_index + 1u)
        );
        if ((low_differs | high_differs) && first == ~0u) {
            first = 2u * i + (low_differs ? 0u : 1u);
        }
        bad += low_differs + high_differs;
    }
    host_push(
        dispatcher,
        request,
        KM_EVENT_PONG | ((bad < 0xFFFFFFu ? bad : 0xFFFFFFu) << 8),
        first
    );
}
#endif

/* Reads source `source`'s line: kernel commands act, bad ones are rejected, others are
 * submitted or wait. `argument` is a0 (microseconds, tag, offset, request, address). */
KMD_HOT kmd_take_t take(kmd_t* dispatcher, uint32_t source, uint64_t now) {
    kmd_ring_t* ring = &dispatcher->ring[source];
    uint64_t* line = ring->line;
    const uint32_t request = ring->next, host = source == KMD_SRC_HOST;
    uint32_t header = (uint32_t)(line[0] >> 32);
    const uint32_t handler = header & 0xFFu, flags = (header >> 8) & 0xFFu;
    const uint32_t group = (uint32_t)line[1], argument = (uint32_t)(line[1] >> 32);
    uint32_t first = group & 0xFFFFu, count = group >> 16;
    const uint32_t queued = handler != KM_HANDLER_EXIT && (flags & KM_FLAG_QUEUE);
    if (handler == KM_HANDLER_DEADLINE && argument != 0) {
        ring_consumed(ring);
        ring->t_command = (uint64_t)argument * 50u;
        return KMD_TAKEN;
    }
    if (handler == KM_HANDLER_STOP && host) {
        ring_consumed(ring);
        if (dispatcher->busy) {
            result_error(
                dispatcher,
                KMD_ERROR_STOP_BUSY,
                0,
                dispatcher->sequence,
                request,
                (uint32_t)dispatcher->busy
            );
        }
        return KMD_FINISH;
    }
    if (handler == KMD_HANDLER_POST && source == KMD_SRC_ENGINE) {
        if (engine_post(dispatcher, line, 0)) {
            return KMD_NO_ROOM;
        }
        ring_consumed(ring);
        dispatcher->last_event_mtime = now;
        return KMD_TAKEN;
    }
    if (handler == KMD_HANDLER_DROP && source == KMD_SRC_ENGINE) {
        ring_consumed(ring);
        queue_drop(dispatcher, argument);
        return KMD_TAKEN;
    }
    if (
        handler == KMD_HANDLER_ANSWER && source >= KMD_SRC_WORK && source < KMD_SRC_LINK
    ) {
        ring_consumed(ring);
        work_answer(
            dispatcher, source - KMD_SRC_WORK, argument, (uint32_t)(line[2] >> 32), now
        );
        return KMD_TAKEN;
    }
    /* Every other command ends in one event: it waits until that event has room. */
    if (!ring_room(ring)) {
        return KMD_NO_ROOM;
    }
    if (handler == KM_HANDLER_PING) {
        ring_consumed(ring);
        ring_push(ring, request, KM_EVENT_PONG, 0);
        dispatcher->last_event_mtime = now;
        return KMD_TAKEN;
    }
    if (host && handler == KMD_HANDLER_RESET && count == KMD_RESET_KEEP) {
        ring_consumed(ring);
        const uint32_t kept = (
            reset_keep(dispatcher, argument, (uint32_t)(line[2] >> 32))
        );
        host_push(dispatcher, request, KM_EVENT_PONG | kept << 8, 0);
        dispatcher->last_event_mtime = now;
        return KMD_TAKEN;
    }
    if (
        host && handler == KMD_HANDLER_RESET && count == 1u &&
        !group_outside(dispatcher, first, count)
    ) {
        ring_consumed(ring);
        reset_command(dispatcher, request, first, argument, (uint32_t)(line[2] >> 32));
        return KMD_TAKEN;
    }
    if (
        host && handler == KMD_HANDLER_INSPECT &&
        !group_outside(dispatcher, first, count)
    ) {
        if (dispatcher->inspect.active) {
            return KMD_HELD; /* one inspect at a time */
        }
        ring_consumed(ring);
        ring->owed++; /* its PONG, once the last core is read */
        inspect_begin(
            dispatcher,
            KMD_INSPECT_HOST,
            request,
            first,
            count,
            0,
            dispatcher->sequence,
            now
        );
        return KMD_TAKEN;
    }
    if (host && (handler == KM_HANDLER_RUN || handler == KM_HANDLER_CANCEL)) {
        /* a2 != 0: a graph's RUN; a CANCEL goes to the graph or stream with its
         * request. */
        int wait = (
            handler == KM_HANDLER_RUN ? -1 : graph_cancel(dispatcher, request, argument)
        );
        if (handler == KM_HANDLER_RUN && (uint32_t)line[3] != 0) {
            wait = graph_run(dispatcher, request, line);
        } else if (wait < 0) {
            wait = (
                handler == KM_HANDLER_RUN
                    ? engine_run(
                        dispatcher, request, argument, (uint32_t)(line[2] >> 32)
                    )
                    : engine_cancel(dispatcher, request, argument)
            );
        }
        if (wait) {
            return KMD_NO_ROOM;
        }
        ring_consumed(ring);
        dispatcher->last_event_mtime = now;
        return KMD_TAKEN;
    }
#ifdef KMD_VERIFY_LINES
    if (host && handler == KMD_HANDLER_CHECK_UPLOAD) {
        ring_consumed(ring);
        upload_check(
            dispatcher, request, argument, (uint32_t)(line[2] >> 32), (uint32_t)line[3]
        );
        dispatcher->last_event_mtime = now;
        return KMD_TAKEN;
    }
    if (host && handler == KMD_HANDLER_TRAP) {
        ring_consumed(ring);
        const uint64_t loaded = *(volatile const uint64_t*)(uintptr_t)argument;
        ring_push(
            ring, request, KM_EVENT_PONG, (uint32_t)loaded
        ); /* only if the load did not trap */
        return KMD_TAKEN;
    }
#endif
#ifdef KMD_STALL_DEBUG
    if (host && handler == KMD_HANDLER_STALL) {
        ring_consumed(ring);
        ring_push(ring, request, KM_EVENT_PONG, 0);
        ring_flush(ring);
        /* No pass for a0 ms: the watcher's interrupt ends it first, or the runtime's
         * stop */
        const uint64_t until = now + (uint64_t)argument * 50000u;
        while (
            x280_mtime() < until &&
            !x280_hart_stop_requested(dispatcher->core, dispatcher->run)
        ) {
        }
        return KMD_TAKEN;
    }
#endif
    /* A group or thread on another island: its parts cross (link.h). A command from
     * there, or the host's part of a dynamic run (its messages cross), runs here. */
    if (
        dispatcher->link_on && source != KMD_SRC_LINK && !queued &&
        handler != KM_HANDLER_EXIT && link_reaches_out(dispatcher, group) &&
        !(host && KM_IS_DYNAMIC(dispatcher->messages.handler, handler))
    ) {
        return link_cross(dispatcher, source, now);
    }
    /* An engine's queued job carries its stream's tag in `first`: never a cluster. */
    if (
        first >= KM_GROUP_CLUSTER_BASE && handler <= KM_HANDLER_USER_MAX &&
        !(queued && source == KMD_SRC_ENGINE)
    ) {
        return work_command(dispatcher, source, now); /* to a work cluster's thread */
    }
    if (!queued && handler != KM_HANDLER_EXIT) { /* flat ranks: this island's part */
        const uint32_t part = island_part(dispatcher, group);
        first = part & 0xFFFFu;
        count = part >> 16; /* none here: count 0, rejected as outside */
    }
    uint32_t reject = 0;
    if (handler == KM_HANDLER_DEADLINE) {
        reject = KMD_REJECT_DEADLINE;
    } else if (
        (
            handler > KM_HANDLER_USER_MAX && handler != KM_HANDLER_EXIT &&
            !(
                (host || (source >= KMD_SRC_WORK && source < KMD_SRC_LINK)) &&
                (handler == KM_HANDLER_LOAD || handler == KM_HANDLER_COPY)
            ) &&
            !(
                host &&
                (
                    handler == KM_HANDLER_PUSH || handler == KM_HANDLER_PULL ||
                    handler == KM_HANDLER_PAGES
                )
            )
        ) ||
        (!host && handler == KM_HANDLER_EXIT)
    ) {
        reject = KMD_REJECT_HANDLER;
    } else if (flags & 1u) {
        reject = KMD_REJECT_FLAGS;
    } else if (dispatcher->exited) {
        reject = KMD_REJECT_EXITED;
    } else if (
        handler != KM_HANDLER_EXIT &&
        group_outside(dispatcher, queued ? 0 : first, count)
    ) {
        reject = KMD_REJECT_GROUP;
    } else if (queued && KM_IS_DYNAMIC(dispatcher->messages.handler, handler)) {
        reject = KMD_REJECT_MESSAGES;
    } else if (
        dispatcher->programs && handler != KM_HANDLER_NOP &&
        handler != KM_HANDLER_EXIT && handler != KM_HANDLER_COPY &&
        (
            handler == KM_HANDLER_LOAD
                ? (header >> 24) >= KMD_PROGRAMS_MAX || queued
                : !program_held(
                    dispatcher,
                    header >> 24,
                    queued ? 0 : first,
                    queued ? dispatcher->cores : count
                )
        )
    ) {
        reject = KMD_REJECT_PROGRAM; /* a queued command: every core holds its program
                                      */
    } else if (handler == KM_HANDLER_LOAD && !dispatcher->programs) {
        reject = KMD_REJECT_PROGRAM; /* a run without the table */
    }
    if (reject != 0) {
        ring_consumed(ring);
        ring_push(ring, request, KM_EVENT_REJECTED | (reject << 8), 0);
        dispatcher->last_event_mtime = now;
        dispatcher->rejected++;
        return KMD_TAKEN;
    }
    if (queued) {
        /* Into the queue; when full, the host's is rejected, others' rejected busy. */
        if (dispatcher->queue->length == KM_QUEUE_JOBS) {
            if (!host) {
                return KMD_HELD;
            }
            ring_consumed(ring);
            ring_push(ring, request, KM_EVENT_REJECTED | (KMD_REJECT_QUEUE << 8), 0);
            dispatcher->last_event_mtime = now;
            dispatcher->rejected++;
            return KMD_TAKEN;
        }
        queue_put(dispatcher, line, source, request, count, host ? 0u : first);
        ring_consumed(ring);
        return KMD_TAKEN;
    }
    if (handler == KM_HANDLER_EXIT) {
        /* EXIT goes to every core, once no command is in flight or queued and no stream
         * runs. */
        if (
            dispatcher->busy || dispatcher->stream_count || dispatcher->queued ||
            graph_running(dispatcher)
        ) {
            return KMD_HELD;
        }
        line[1] = (
            (line[1] & ~(uint64_t)0xFFFFFFFFu) | dispatcher->island_first |
            KM_GROUP_ALL(dispatcher->cores)
        );
    } else if (
        (
            (flags & KM_FLAG_EXCLUSIVE) &&
            (ring->before || slots_overlap(dispatcher, first, count))
        ) ||
        messages_held(dispatcher, handler) ||
        (
            handler == KM_HANDLER_LOAD &&
            (dispatcher->queued || slots_overlap(dispatcher, first, count))
        )
    ) {
        return KMD_HELD; /* exclusive: every job queued before it placed, no overlap */
    }
    const uint32_t slot = slot_free(dispatcher);
    if (slot == KM_RING_SLOTS) {
        return KMD_HELD; /* another source took the last free slot */
    }
    line[0] = (line[0] & 0xFFFFFFFFu) | (uint64_t)header << 32;
    command_submit(
        dispatcher, line, slot, request, source, now, ring->t_command, KMD_TRACE_SUBMIT
    );
    messages_submitted(dispatcher, line, slot, request, source);
    if (handler == KM_HANDLER_LOAD) {
        program_move(
            dispatcher, header >> 24, first, count
        ); /* the cores take it in order */
    }
    ring_consumed(ring);
    dispatcher->exited = handler == KM_HANDLER_EXIT;
    return KMD_TAKEN;
}

/* A command of the engine or a work cluster that cannot go now (no free slot, an
 * overlapping exclusive group, a full queue): rejected at once, KMD_EVENT_WAKE owed. */
static int source_busy(kmd_t* dispatcher, uint32_t source, uint64_t now) {
    kmd_ring_t* ring = &dispatcher->ring[source];
    if (!(dispatcher->waits & (1u << source))) {
        ring->owed++;
        if (!ring_room(ring)) {
            ring->owed--;
            return 0; /* no room for the rejection and the wake: it waits for room */
        }
        dispatcher->waits |= 1u << source;
    }
    const uint32_t header = (uint32_t)(ring->line[0] >> 32);
    if (source == KMD_SRC_ENGINE && ((header >> 8) & KM_FLAG_QUEUE)) {
        /* A rejected job counts as taken in, as the engine counts every job sent. */
        *(volatile uint32_t*)(
            dispatcher->ering + KMD_ERING_TAKEN
        ) = ++dispatcher->queue->taken;
    }
    ring_push(ring, ring->next, KM_EVENT_REJECTED | (KMD_REJECT_BUSY << 8), 0);
    x280_fence_ow(); /* the rejection lands before the consumed count passes it */
    ring_consumed(ring);
    dispatcher->last_event_mtime = now;
    dispatcher->rejected++;
    return 1;
}

/* Source `source`'s turn in a pass: at most one command, the host's only while a slot
 * is free. Returns 1 when the command ends the loop. */
KMD_HOT int source_turn(kmd_t* dispatcher, uint32_t source, uint32_t* activity) {
    kmd_ring_t* ring = &dispatcher->ring[source];
    const uint32_t host = source == KMD_SRC_HOST;
    if (ring->line_state == 0) {
        if (dispatcher->busy == dispatcher->full && host) {
            dispatcher->full_passes += host;
            return 0;
        }
        if (!ring_take(ring)) {
            dispatcher->idle_passes += host;
            return 0;
        }
        ring->line_state = 1;
        ring->before = dispatcher->queue->used;
        ++*activity;
#ifdef KMD_VERIFY_LINES
        if (host) {
            verify_line(dispatcher);
        }
#endif
    }
    const uint64_t now = x280_mtime();
    if (dispatcher->first_command_mtime == 0) {
        dispatcher->first_command_mtime = now;
    }
    const kmd_take_t outcome = take(dispatcher, source, now);
    if (outcome == KMD_FINISH) {
        return 1;
    }
    if (outcome == KMD_TAKEN) {
        ring->line_state = 0;
    } else if (
        outcome == KMD_HELD && !host && source != KMD_SRC_LINK &&
        source_busy(dispatcher, source, now)
    ) {
        ring->line_state = 0; /* only the host's and the other island's wait */
    } else if (ring->line_state == 1) {
        ring->line_state = 2;
        if (outcome == KMD_HELD) {
            dispatcher->held++;
        } else {
            dispatcher->no_room++;
        }
    }
    return 0;
}

/* The loop: sweep, one command of each source in turn (host, engine), the queue's turn.
 * Returns when a STOP command, the runtime's stop or an error ends it. */
static void loop(kmd_t* dispatcher) {
    const volatile kmd_args_t* arguments = dispatcher->args;
    result_where(dispatcher, KMD_WHERE_LOOP);
    uint32_t spins = 0, activity = 0, seen_activity = 0;
    uint64_t active_since = x280_mtime();
    kmd_ring_t* host = &dispatcher->ring[KMD_SRC_HOST];
    kmd_ring_t* engine = &dispatcher->ring[KMD_SRC_ENGINE];
    for (;;) {
        dispatcher->passes++;
        if (dispatcher->busy) {
            slots_sweep(dispatcher);
        }
        if (dispatcher->inspect.active) {
            inspect_turn(dispatcher);
        }
        if (dispatcher->messages.active) {
            messages_turn(dispatcher);
        }
        if (source_turn(dispatcher, KMD_SRC_HOST, &activity)) {
            return;
        }
        if (dispatcher->engine_on) {
            source_turn(dispatcher, KMD_SRC_ENGINE, &activity);
            ring_flush(engine);
            dispatcher->engine_on = engine_active(dispatcher);
        }
        if (dispatcher->queued != 0) {
            queue_turn(dispatcher, x280_mtime());
        }
        /* Work ring w is looked at on passes w mod 4, at most one ring a pass: one
         * look costs the idle pass 31 ns. */
        const uint32_t work_ring = (uint32_t)(dispatcher->passes & 3u);
        if (
            work_ring < KMD_WORK_CLUSTERS &&
            dispatcher->ring[KMD_SRC_WORK + work_ring].commands != 0
        ) {
            source_turn(dispatcher, KMD_SRC_WORK + work_ring, &activity);
            ring_flush(&dispatcher->ring[KMD_SRC_WORK + work_ring]);
        }
        if (dispatcher->link_on) {
            link_turn(dispatcher, x280_mtime());
            source_turn(dispatcher, KMD_SRC_LINK, &activity);
        }
        ring_flush(host);
        if (++spins != KMD_POLL_EVERY) {
            continue;
        }
        spins = 0;
        dispatcher->x280_cores[KMD_H_MARKER0] = (uint32_t)(dispatcher->passes >> 10);
        if (x280_hart_stop_requested(dispatcher->core, dispatcher->run)) {
            result_error(
                dispatcher,
                KMD_ERROR_STOPPED,
                0,
                dispatcher->sequence,
                0,
                (uint32_t)dispatcher->busy
            );
            return;
        }
        const uint64_t now = x280_mtime();
        if (dispatcher->thread_busy) {
            work_late(dispatcher, now);
        }
        if (dispatcher->link_on) {
            link_late(dispatcher, now);
            link_count(dispatcher);
        }
        const uint32_t late = slots_late(dispatcher, now);
        if (late != KM_RING_SLOTS && inspect_may_give_up(dispatcher, late)) {
            /* The run goes on: an inspect of the group, then the abort (inspect.h);
             * while another inspect is read, the next poll looks again. */
            if (!dispatcher->inspect.active) {
                inspect_timeout(dispatcher, late, now);
            }
        } else if (late != KM_RING_SLOTS) {
            /* The command's source gets the timeout event; the engine turns it into its
             * stream's error, which engine_stop forwards. */
            const uint32_t took = (uint32_t)(now - dispatcher->slot_submitted[late]);
            const uint32_t source = dispatcher->slot_source[late];
            result_error(
                dispatcher,
                KMD_ERROR_DEADLINE,
                late,
                dispatcher->slot_sequence[late],
                dispatcher->slot_request[late],
                took
            );
            dispatcher->x280_cores[KMD_H_SOURCE] = source;
            const uint32_t auxiliary = (
                late |
                (dispatcher->slot_sequence[late] & KM_EVENT_SEQUENCE_MASK)
                    << KM_EVENT_SEQUENCE_SHIFT
            );
            ring_push(
                &dispatcher->ring[source],
                dispatcher->slot_request[late],
                KM_EVENT_TIMEOUT | auxiliary << 8,
                took
            );
            return;
        }
        if (
            activity != seen_activity || dispatcher->busy || dispatcher->stream_count ||
            dispatcher->queued || dispatcher->thread_busy ||
            graph_running(dispatcher) || dispatcher->inspect.active ||
            (dispatcher->link_on && link_active(dispatcher))
        ) {
            seen_activity = activity;
            active_since = now;
        } else if (arguments->t_idle != 0 && now - active_since > arguments->t_idle) {
            result_error(dispatcher, KMD_ERROR_IDLE, 0, dispatcher->sequence, 0, 0);
            return;
        }
    }
}

/* The stop flag goes out only while a slot is busy or before EXIT was submitted: after
 * a completed EXIT the cores have left and their L1 is no longer the program's. */
static void finish(kmd_t* dispatcher) {
    watch_off(dispatcher->x280_cores);
    result_where(dispatcher, KMD_WHERE_FINISH);
    uint32_t abandoned = dispatcher
                             ->queued; /* no DONE event for these: dropped or cut off */
    for (uint32_t slot = 0; slot < KM_RING_SLOTS; slot++) {
        abandoned += (uint32_t)(dispatcher->busy >> slot) & 1u;
    }
    abandoned += work_abandoned(dispatcher);
    dispatcher->x280_cores[KMD_H_ABANDONED] = abandoned;
    if (dispatcher->control[0] != 0 && (dispatcher->busy || !dispatcher->exited)) {
        grid_stop(dispatcher);
    }
    engine_stop(dispatcher);
    work_close(dispatcher);
    inspect_close(dispatcher);
    if (dispatcher->link_on) {
        link_count(dispatcher);
    }
    link_close(dispatcher);
    const uint64_t now = x280_mtime();
    result_finish(dispatcher, now);
#ifdef KMD_TRACE
    const uint32_t header[4] = {
        KMD_TRACE_MARK, dispatcher->traced, dispatcher->untraced, 0
    };
    if (dispatcher->upload_bytes >= 64u) {
        kmd_store_chunk((volatile uint8_t*)dispatcher->upload, header);
    }
#endif
    ring_flush(&dispatcher->ring[KMD_SRC_HOST]);
    events_word(dispatcher, KMD_HOST_ERROR, dispatcher->error);
    events_word(
        dispatcher,
        KMD_HOST_STATE,
        dispatcher->error ? KMD_STATE_FAILED : KMD_STATE_DONE
    );
    grid_close(dispatcher);
    events_close(dispatcher);
    const uint64_t ticks = (
        dispatcher->first_command_mtime ? now - dispatcher->first_command_mtime : 0
    );
    const uint64_t core = dispatcher->core;
    const uint64_t result = dispatcher->error ? ~(uint64_t)dispatcher->error : ticks;
    kmd_flush_lines(dispatcher, KMD_STATE_BYTES);
    x280_hart_set_result(core, result);
}

/* After an exception on core `core` (trap.h). Core 0 finishes as an abort does from LIM
 * and the argument block alone: the state's pointers may be what failed. */
__attribute__((used, noinline)) void kmd_trapped(
    uint64_t core, uint64_t cause, uint64_t epc, uint64_t tval
) {
    const uint64_t run_value = x280_hart_control(core)->run_value;
    const volatile kmd_args_t* arguments = (
        (const volatile kmd_args_t*) x280_hart_args(core)
    );
    x280_hart_set_result(core, ~(uint64_t)KMD_ERROR_TRAP);
    if (arguments == 0 || !args_valid(arguments, run_value) || core > 1) {
        return;
    }
    volatile uint8_t* lim = (volatile uint8_t*)arguments;
    volatile uint32_t* x280_cores = (volatile uint32_t*)(lim + KMD_LIM_CORES);
    if (core == 0) {
        watch_off(x280_cores);
    }
    if (core == 1) {
        x280_cores[KMD_H_TRAP_CAUSE] = (uint32_t)cause;
        x280_cores[KMD_H_TRAP_PC] = (uint32_t)epc;
        x280_fence_ow();
        x280_cores[KMD_H_ENGINE] = KMD_ENGINE_TRAPPED;
        return;
    }
    /* A fresh state on the stack, zeroed through a volatile pointer as in run(). */
    kmd_t fresh;
    volatile uint64_t* raw = (volatile uint64_t*)&fresh;
    for (uint32_t i = 0; i < sizeof(kmd_t) / sizeof(uint64_t); i++) {
        raw[i] = 0;
    }
    kmd_t* dispatcher = &fresh;
    dispatcher->core = core;
    dispatcher->run = run_value;
    dispatcher->args = arguments;
    dispatcher->lim = lim;
    dispatcher->result = (volatile uint32_t*)(lim + KMD_LIM_RESULT);
    dispatcher->x280_cores = x280_cores;
    dispatcher->panels = (uint32_t)arguments->panels;
    const uint32_t where = dispatcher->result[KMD_R_WHERE];
    result_error(
        dispatcher, KMD_ERROR_TRAP, (uint32_t)cause, (uint32_t)epc, 0, (uint32_t)tval
    );
    /* The old state's words, read but never followed: whether EXIT left the grid. */
    volatile uint8_t* state = (
        (volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + arguments->state_ddr)
    );
    const volatile kmd_t* old = (const volatile kmd_t*)state;
    const uint32_t control = ((volatile const uint32_t*)(lim + KMD_LIM_MAILBOX))[1];
    if (
        where >= KMD_WHERE_READY && (old->busy || !old->exited) &&
        grid_open(dispatcher, control) == KMD_ERROR_NONE
    ) {
        grid_stop(dispatcher);
    }
    /* The engine: told not to start, or to stop; waited for as at the finish. */
    volatile uint32_t* go = (volatile uint32_t*)(state + KMD_STATE_RING + KMD_ERING_GO);
    if (x280_cores[KMD_H_ENGINE] == KMD_ENGINE_WAITING) {
        go[2] = 2u;
        x280_fence_ow();
        *(volatile uint64_t*)go = run_value;
    }
    *(volatile uint32_t*)(state + KMD_STATE_RING + KMD_ERING_STOP) = 1u;
    x280_fence_ow();
    const uint64_t deadline = x280_mtime() + KMD_ENGINE_STOP_TICKS;
    while (
        (
            x280_cores[KMD_H_ENGINE] == KMD_ENGINE_WAITING ||
            x280_cores[KMD_H_ENGINE] == KMD_ENGINE_RUNNING
        ) &&
        x280_mtime() <= deadline
    ) {
        x280_fence_ir();
    }
    if (where >= KMD_WHERE_HELLO && events_open(dispatcher) == KMD_ERROR_NONE) {
        events_word(dispatcher, KMD_HOST_ERROR, KMD_ERROR_TRAP);
        events_word(dispatcher, KMD_HOST_STATE, KMD_STATE_FAILED);
    }
    grid_close(dispatcher);
    events_close(dispatcher);
    for (uint32_t cluster = 0; cluster < KMD_WORK_CLUSTERS; cluster++) {
        noc_clear_tlb_2m(KMD_WIN_WORK + cluster);
    }
    noc_clear_tlb_2m(KMD_WIN_INSPECT);
    noc_clear_tlb_2m(KMD_WIN_LINK);
    kmd_flush_lines(state, KMD_STATE_BYTES);
    x280_fence_ow();
    result_word(dispatcher, KMD_R_STATE, KMD_STATE_FAILED);
}

/* Core 0 after the watcher's interrupt, on run()'s frame: a line consumed before the
 * stall freed, BACK to the host, then the loop again on the run's state. */
__attribute__((used, noinline)) void kmd_resumed(kmd_t* dispatcher, uint64_t epc) {
    *watch_msip(0) = 0u;
    x280_fence_iorw();
    for (uint32_t source = 0; source < KMD_SOURCES; source++) {
        kmd_ring_t* ring = &dispatcher->ring[source];
        if (ring->line_state != 0 && (uint32_t)ring->line[0] != ring->next) {
            ring->line_state = 0;
        }
    }
    noc_clear_tlb_2m(KMD_WIN_RESET);
    noc_clear_tlb_2m(KMD_WIN_RESET_REGISTER);
    kmd_ring_t* host = &dispatcher->ring[KMD_SRC_HOST];
    if (host->events != 0 && ring_room(host)) {
        const uint32_t took = (uint32_t)(x280_mtime() - dispatcher->stall_mtime);
        ring_push(host, (uint32_t)epc, KMD_EVENT_STALL | KMD_STALL_BACK << 8, took);
        ring_flush(host);
    }
    dispatcher->x280_cores[KMD_H_STALL_PC] = (uint32_t)epc;
    dispatcher->x280_cores[KMD_H_BACK]++;
    watch_on(dispatcher);
    loop(dispatcher);
    finish(dispatcher);
}

/* Core 0's kernel; kept out of main so main's prologue touches no FP register. */
static __attribute__((noinline)) void run(uint64_t core) {
    uint64_t host_line[16], engine_line[16], work_lines[8u * KMD_WORK_CLUSTERS + 8u];
    const uint64_t run_value = x280_hart_control(core)->run_value;
    const uint64_t start_mtime = x280_mtime();
    const volatile kmd_args_t* arguments = (
        (const volatile kmd_args_t*) x280_hart_args(core)
    );
    if (arguments == 0 || !args_valid(arguments, run_value)) {
        x280_hart_set_result(core, ~(uint64_t)KMD_ERROR_ARGS);
        return;
    }
    kmd_t* dispatcher = (
        (kmd_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + arguments->state_ddr)
    );
    /* The whole state area, engine's too, zeroed through a volatile pointer: a plain
     * loop may become a memset call, and the image has no C library. */
    volatile uint64_t* raw = (volatile uint64_t*)dispatcher;
    for (uint32_t i = 0; i < KMD_STATE_BYTES / sizeof(uint64_t); i++) {
        raw[i] = 0;
    }
    dispatcher->core = core;
    dispatcher->run = run_value;
    dispatcher->start_mtime = start_mtime;
    dispatcher->args = arguments;
    dispatcher->lim = (volatile uint8_t*)arguments;
    dispatcher->result = (volatile uint32_t*)(dispatcher->lim + KMD_LIM_RESULT);
    dispatcher->x280_cores = (volatile uint32_t*)(dispatcher->lim + KMD_LIM_CORES);
    dispatcher->upload = dispatcher->lim + KMD_LIM_UPLOAD;
    dispatcher->upload_bytes = (uint32_t)arguments->upload_bytes;
    dispatcher->panels = (uint32_t)arguments->panels;
    dispatcher->depth = (uint32_t)arguments->depth;
    dispatcher->full = ~0ull >> (64u - dispatcher->depth);
    dispatcher->cores = (uint32_t)arguments->cores;
    dispatcher->island_first = (uint32_t)arguments->island & 0xFFFFu;
    dispatcher->island_count = (uint32_t)(arguments->island >> 16) & 0xFFFFu;
    dispatcher->island_base = (uint32_t)(arguments->island >> 32) & 0xFFFFu;
    dispatcher->messages.handler = (uint32_t)arguments->messages_handler;
    dispatcher->messages.x280_cluster = (
        KMD_X280_RANKS_CLUSTER(arguments->messages_handler)
    );
    dispatcher->messages.x280_count = KMD_X280_RANKS_COUNT(arguments->messages_handler);
    dispatcher->programs = (uint32_t)arguments->programs;
    for (uint32_t program = 0; program < KMD_PROGRAMS_MAX; program++) {
        dispatcher->holds[program][0] = dispatcher->holds[program][1] = 0;
    }
    for (uint32_t core = 0; dispatcher->programs && core < dispatcher->cores; core++) {
        const uint32_t program = dispatcher->lim[KMD_LIM_PROGRAMS + core];
        if (program < KMD_PROGRAMS_MAX) {
            dispatcher->holds[program][core >> 6] |= 1ull << (core & 63u);
        }
    }
    dispatcher->ering = (volatile uint8_t*)dispatcher + KMD_STATE_RING;
    dispatcher->queue = (kmd_queue_t*)((uint8_t*)dispatcher + KMD_STATE_QUEUE);
    kmd_ring_t* host_ring = &dispatcher->ring[KMD_SRC_HOST];
    host_ring->commands = dispatcher->lim + KMD_LIM_HRING;
    host_ring->read = (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_READ);
    host_ring->depth = (uint32_t)arguments->host_depth;
    host_ring->event_depth = (uint32_t)arguments->event_depth;
    host_ring->next = 1;
    host_ring->t_command = arguments->t_command;
    host_ring->line = (uint64_t*)(((uintptr_t)host_line + 63u) & ~(uintptr_t)63u);
    result_word(dispatcher, KMD_R_STATE, KMD_STATE_BOOT);
    uint64_t* aligned_work_lines = (
        (uint64_t*)(((uintptr_t)work_lines + 63u) & ~(uintptr_t)63u)
    );
    if (
        boot(dispatcher) == KMD_ERROR_NONE &&
        work_open(dispatcher, aligned_work_lines) != KMD_ERROR_NONE
    ) {
        result_error(dispatcher, KMD_ERROR_WINDOW, 0, 0, 0, 0);
    }
    if (dispatcher->error == KMD_ERROR_NONE) {
        const uint32_t linked = link_open(dispatcher);
        if (linked != KMD_ERROR_NONE) {
            result_error(dispatcher, linked, 0, 0, 0, 0);
        }
    }
    if (dispatcher->error == KMD_ERROR_NONE) {
        engine_open(
            dispatcher, (uint64_t*)(((uintptr_t)engine_line + 63u) & ~(uintptr_t)63u)
        );
        engine_go(dispatcher, 1u);
        result_word(
            dispatcher,
            KMD_R_BOOT_TICKS,
            (uint32_t)(x280_mtime() - dispatcher->start_mtime)
        );
        result_word(dispatcher, KMD_R_STATE, KMD_STATE_READY);
        events_ready(dispatcher);
        events_word(dispatcher, KMD_HOST_STATE, KMD_STATE_READY);
        /* For the watcher's interrupt (trap.h): this frame's sp and the state. */
        __asm__ volatile(
            "csrr t0, mscratch\n"
            "sd sp, 112(t0)\n"
            "sd %0, 120(t0)\n" ::"r"(dispatcher)
            : "t0", "memory"
        );
        watch_on(dispatcher);
        loop(dispatcher);
    } else {
        engine_go(dispatcher, 2u);
    }
    finish(dispatcher);
}

/* Called by main (trap.h) on every core. */
__attribute__((used, noinline)) int kmd_main(uint64_t core_id) {
    /* Floating point and vector on first, on every core: the harness leaves them off
     * and the first such instruction would trap. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    if (core_id == 0) {
        run(core_id);
    } else if (core_id == 1) {
        const uint64_t run_value = x280_hart_control(core_id)->run_value;
        const volatile kmd_args_t* arguments = (
            (const volatile kmd_args_t*) x280_hart_args(core_id)
        );
        if (arguments != 0 && args_valid(arguments, run_value)) {
            eng_run(core_id, arguments, run_value);
        }
    } else if (core_id == 2) {
        const uint64_t run_value = x280_hart_control(core_id)->run_value;
        const volatile kmd_args_t* arguments = (
            (const volatile kmd_args_t*) x280_hart_args(core_id)
        );
        if (arguments != 0 && args_valid(arguments, run_value)) {
            watch(core_id, arguments, run_value);
        }
    }
    return 0;
}
