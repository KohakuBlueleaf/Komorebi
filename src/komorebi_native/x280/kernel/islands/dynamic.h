/* Dynamic runs: two passes over the RUN group's status lines; when all are idle and
 * unchanged and sent equals consumed, the run's end command goes to the group. */

#pragma once

#include "../sources/completion.h"
#include "../kernel.h"

/* waves.h, after link.h: a run across islands */
static void messages_islands(kmd_t* dispatcher);
static void messages_entry(kmd_t* dispatcher, uint32_t id);
static int messages_waves(kmd_t* dispatcher);

/* The run line into the X280 ranks' thread area: one 16-byte store, then a fence. */
static inline void messages_run_store(kmd_thread_t* thread) {
    kmd_store_chunk(
        thread->lines + KMD_WORK_IN_DEPTH * 64u, (const uint32_t*)thread->run_line
    );
    x280_fence_ow();
    thread->run_unstored = 0;
}

static inline int messages_spans(const kmd_t* dispatcher);
static uint32_t messages_x280_counts(kmd_t* dispatcher);

/* The X280 ranks' cluster learns the run's start or end (KM_X280_RUN_*): stored now
 * if its thread serves, else once it does (its start-up writes the area over). */
static inline void messages_run_line(kmd_t* dispatcher, uint32_t state) {
    kmd_messages_t* run = &dispatcher->messages;
    if (run->x280_count == 0) {
        return;
    }
    kmd_thread_t* thread = &dispatcher->thread[run->x280_cluster - 1u];
    thread->run_line[KM_X280_RUN_SEQUENCE] = run->sequence;
    thread->run_line[KM_X280_RUN_STATE] = state;
    thread->run_line[KM_X280_RUN_GROUP] = run->group;
    thread->run_line
        [KM_X280_RUN_HANDLER] = (run->run_handler | messages_x280_counts(dispatcher));
    if (thread->serving && thread->lines != 0) {
        messages_run_store(thread);
    } else {
        thread->run_unstored = 1;
    }
}

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
    run->part = island_part(dispatcher, run->group);
    run->request = request;
    run->source = source;
    run->run_handler = (uint32_t)(line[0] >> 32) & 0xFFu;
    run->id = run->round = run->ended = run->have_last = 0; /* a ROUND or START kept */
    run->round_number = 0;
    run->next_round = 0;
    if (run->deliver && dispatcher->link_on) {
        messages_islands(dispatcher);
    }
    messages_run_line(dispatcher, KM_X280_RUN_START);
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
    messages_run_line(dispatcher, KM_X280_RUN_END);
}

/* 1 when the run's group spans other islands and this island has link cores: link
 * core 0 serves the run's messages across and its line counts with the group's. */
static inline uint32_t messages_link(const kmd_t* dispatcher) {
    const kmd_messages_t* run = &dispatcher->messages;
    return (
        !run->deliver && dispatcher->island_count != 0 &&
        (run->part >> 16) < (run->group >> 16) &&
        dispatcher->cores > dispatcher->island_base + dispatcher->island_count
    );
}

/* The index of the run's line `index` of `lines`: the group's cores on this island,
 * link core 0 (base + ranks) when it serves the run, then the X280 ranks. */
static inline uint32_t messages_line(const kmd_t* dispatcher, uint32_t index) {
    const kmd_messages_t* run = &dispatcher->messages;
    const uint32_t first = run->part & 0xFFFFu, count = run->part >> 16;
    const uint32_t link = messages_link(dispatcher);
    if (index < count) {
        return first + index;
    }
    return (
        index < count + link
            ? dispatcher->island_base + dispatcher->island_count
            : KM_X280_RANK_CORE(index - count - link)
    );
}

/* The two passes: 1 every line idle and unchanged, its sums in `sent_out` and
 * `consumed_out`; 0 not yet. Counter tiles and X280 ranks are read as other cores. */
static inline int messages_sample(
    kmd_t* dispatcher, uint64_t* sent_out, uint64_t* consumed_out
) {
    kmd_messages_t* run = &dispatcher->messages;
    *sent_out = *consumed_out = 0;
    const uint32_t lines_count = (
        (run->part >> 16) + messages_link(dispatcher) + run->x280_count
    );
    volatile const uint32_t* lines = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_MESSAGES)
    );
    dispatcher->x280_cores[KMD_H_MESSAGES_CHECKS] = ++run->checks;
    x280_fence_ir();
    for (uint32_t index = 0; index < lines_count; index++) {
        const uint32_t core = messages_line(dispatcher, index);
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
    for (uint32_t index = 0; index < lines_count; index++) {
        const uint32_t core = messages_line(dispatcher, index);
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
    *sent_out = sent;
    *consumed_out = consumed;
    return 1;
}

/* Quiet on this island alone: every line idle and unchanged, sent equal to consumed. */
static inline int messages_quiet(kmd_t* dispatcher) {
    uint64_t sent = 0, consumed = 0;
    return messages_sample(dispatcher, &sent, &consumed) && sent == consumed;
}

/* Whether the run's group spans islands and its messages cross as deliveries. */
static inline int messages_spans(const kmd_t* dispatcher) {
    const kmd_messages_t* run = &dispatcher->messages;
    return (
        run->deliver && dispatcher->link_on && (run->part >> 16) < (run->group >> 16)
    );
}

/* A run across islands: its X280 ranks' lines and its link cores' (after its ranks)
 * read every pass, so a give-up there ends it before a round's end or the deadline. */
static inline int messages_lines_gave(kmd_t* dispatcher) {
    const kmd_messages_t* run = &dispatcher->messages;
    volatile const uint32_t* lines = (
        (volatile const uint32_t*)(dispatcher->lim + KMD_LIM_MESSAGES)
    );
    volatile const uint32_t* words = (/* the link line, checked at the open */
        (volatile const uint32_t*)(
            dispatcher->lim + KMD_LIM_LINK
        )
    );
    const uint32_t links = words[KMD_L_CORE];
    const uint32_t count = run->x280_count + words[KMD_L_LINKS];
    x280_fence_ir();
    for (uint32_t index = 0; index < count; index++) {
        const uint32_t core = (
            index < run->x280_count
                ? KM_X280_RANK_CORE(index)
                : links + index - run->x280_count
        );
        volatile const uint32_t* line = lines + 4u * core;
        if (
            line[KM_MESSAGES_LINE_RUN] == run->sequence &&
            line[KM_MESSAGES_LINE_STATE] == KM_MESSAGES_GAVE_UP
        ) {
            messages_gave(dispatcher, core);
            return 1;
        }
    }
    return 0;
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
        if (run->id != 0) {
            messages_entry(dispatcher, 0);
        }
        messages_run_line(dispatcher, KM_X280_RUN_END);
        return;
    }
    const int spans = messages_spans(dispatcher);
    if (spans && messages_lines_gave(dispatcher)) {
        return;
    }
    if (spans ? !messages_waves(dispatcher) : !messages_quiet(dispatcher)) {
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
    if (spans) {
        messages_entry(dispatcher, 0); /* no line of this run is on its way */
    }
    messages_run_line(dispatcher, KM_X280_RUN_END);
    dispatcher->x280_cores[KMD_H_MESSAGES_QUIET] = (uint32_t)now;
}
