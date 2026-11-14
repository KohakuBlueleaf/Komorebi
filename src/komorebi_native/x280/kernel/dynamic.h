/* Dynamic runs: two passes over the RUN group's status lines; when all are idle and
 * unchanged and sent equals consumed, the run's end command goes to the group. */

#pragma once

#include "completion.h"
#include "kernel.h"

/* After command_submit: `line` in `slot`, for command `request` of source `source`, may
 * start the dynamic run. */
KMD_HOT void messages_submitted(
    kmd_t* dispatcher,
    const uint64_t* line,
    uint32_t slot,
    uint32_t request,
    uint32_t source
) {
    kmd_messages_t* run = &dispatcher->messages;
    if (!KM_IS_DYNAMIC(run->handler, (uint32_t)(line[0] >> 32) & 0xFFu)) {
        return;
    }
    run->active = 1;
    run->slot = slot;
    run->sequence = (uint32_t)line[0];
    run->group = (uint32_t)line[1];
    run->request = request;
    run->source = source;
}

/* Whether a RUN of handler `handler` must wait: a dynamic run or its end is in flight.
 */
static inline int messages_held(const kmd_t* dispatcher, uint32_t handler) {
    return (
        KM_IS_DYNAMIC(dispatcher->messages.handler, handler) &&
        (dispatcher->messages.active || dispatcher->messages.end)
    );
}

/* A status line that says gave up: its RUN command's deadline becomes now, so the
 * loop's next deadline check ends the session with KMD_ERROR_DEADLINE on it. */
static inline void messages_gave(kmd_t* dispatcher, uint32_t core) {
    dispatcher->x280_cores[KMD_H_MESSAGES_GAVE] = core + 1u;
    dispatcher->slot_deadline[dispatcher->messages.slot] = 0;
    dispatcher->messages.active = 0;
}

/* The two passes: 1 quiet, 0 not yet (a line busy, of an earlier run or changed).
 * Counter tiles are group cores like the rest: idle lines, nothing sent or consumed. */
static inline int messages_quiet(kmd_t* dispatcher) {
    kmd_messages_t* run = &dispatcher->messages;
    const uint32_t first = run->group & 0xFFFFu, end = first + (run->group >> 16);
    volatile const uint32_t* lines = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_MESSAGES)
    );
    dispatcher->x280_cores[KMD_H_MESSAGES_CHECKS] = ++run->checks;
    x280_fence_ir();
    for (uint32_t core = first; core < end; core++) {
        volatile const uint32_t* line = lines + 4u * core;
        if (line[KM_MESSAGES_LINE_RUN] != run->sequence) {
            return 0;
        }
        const uint32_t state = line[KM_MESSAGES_LINE_STATE];
        if (state == KM_MESSAGES_GAVE_UP) {
            messages_gave(dispatcher, core);
            return 0;
        }
        if (state != KM_MESSAGES_IDLE) {
            return 0;
        }
        run->first[2u * core] = line[KM_MESSAGES_LINE_SENT];
        run->first[2u * core + 1u] = line[KM_MESSAGES_LINE_CONSUMED];
    }
    x280_fence_ir();
    uint64_t sent = 0, consumed = 0;
    for (uint32_t core = first; core < end; core++) {
        volatile const uint32_t* line = lines + 4u * core;
        if (
            line[KM_MESSAGES_LINE_RUN] != run->sequence ||
            line[KM_MESSAGES_LINE_STATE] != KM_MESSAGES_IDLE ||
            line[KM_MESSAGES_LINE_SENT] != run->first[2u * core] ||
            line[KM_MESSAGES_LINE_CONSUMED] != run->first[2u * core + 1u]
        ) {
            return 0;
        }
        sent += run->first[2u * core];
        consumed += run->first[2u * core + 1u];
    }
    return sent == consumed;
}

/* A pass during a dynamic run: once quiet and a slot is free, its end command {a0: the
 * run's sequence} goes to the group under the RUN's command and source. */
static inline void messages_turn(kmd_t* dispatcher) {
    kmd_messages_t* run = &dispatcher->messages;
    if (
        !(dispatcher->busy & (1ull << run->slot)) ||
        dispatcher->slot_sequence[run->slot] != run->sequence
    ) {
        run->active = 0; /* the RUN completed or ended otherwise: nothing to end */
        return;
    }
    if (!messages_quiet(dispatcher)) {
        return;
    }
    const uint32_t slot = slot_free(dispatcher);
    if (slot == KM_RING_SLOTS) {
        return;
    }
    const uint64_t now = x280_mtime();
    __attribute__((aligned(64))) uint64_t line[8] = {0};
    line[0] = (uint64_t)KM_HANDLER_MESSAGES_END << 32;
    line[1] = run->group | (uint64_t)run->sequence << 32;
    command_submit(
        dispatcher,
        line,
        slot,
        run->request,
        run->source,
        now,
        dispatcher->ring[run->source].t_command,
        KMD_TRACE_SUBMIT
    );
    run->end = slot + 1u;
    run->active = 0;
    dispatcher->x280_cores[KMD_H_MESSAGES_QUIET] = (uint32_t)now;
}
