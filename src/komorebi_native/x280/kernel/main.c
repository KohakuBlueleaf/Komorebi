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
#include "kernel.h"
#include "queue.h"
#include "result.h"
#include "ring.h"
#include "trap.h"
#include "work_port.h"

/* Waits until `word` holds `value`, the runtime asks for a stop, or `deadline`. */
static kmd_wait_t wait_word(
    const kmd_t* k, volatile const uint32_t* word, uint32_t value, uint64_t deadline
) {
    uint32_t spins = 0;
    for (;;) {
        x280_fence_ir();
        if (*word == value) {
            return KMD_OK;
        }
        const kmd_wait_t w = kmd_poll(k, &spins, deadline);
        if (w != KMD_OK) {
            return w;
        }
    }
}

static int pow2(uint64_t n) {
    return n != 0 && (n & (n - 1u)) == 0;
}

/* Checks the argument block against this run: ring depths are powers of two, so the
 * slot numbering survives the wrap of the 32-bit counters. */
static int args_valid(const volatile kmd_args_t* a, uint64_t run) {
    if (
        x280_hart_args_bytes() < sizeof(kmd_args_t) || a->run_value != run ||
        a->version != KMD_ARGS_VERSION
    ) {
        return 0;
    }
    const uint64_t host_bytes = KMD_HOST_RING + a->event_depth * sizeof(km_event_t);
    return (
        a->depth >= 1 && a->depth <= KM_RING_SLOTS && pow2(a->host_depth) &&
        a->host_depth <= KMD_HOST_DEPTH_MAX && pow2(a->event_depth) &&
        a->columns >= 1 && a->columns <= KMD_READY_MAX && a->cores >= 1 &&
        a->cores <= KM_CORES_MAX && a->panels >= 1 && a->panels <= 2 &&
        a->t_start != 0 && a->t_command != 0 && a->state_ddr != 0 &&
        (a->upload_bytes & 63u) == 0 && a->upload_bytes <= X280_LIM_SIZE &&
        (uintptr_t)a + KMD_LIM_UPLOAD + a->upload_bytes <= X280_LIM_END &&
        (a->state_ddr & 63u) == 0 && (a->host_noc_addr & 63u) == 0 &&
        (a->host_noc_addr & (NOC_2M_WINDOW_STRIDE - 1u)) + host_bytes <=
            NOC_2M_WINDOW_STRIDE &&
        work_args_valid(a) && a->messages_handler <= KM_HANDLER_USER_MAX
    );
}

/* Start-up: host window, hello, panels (so a failed READY wait can stop the grid),
 * READY. Returns KMD_ERROR_NONE or an error. */
static uint32_t boot(kmd_t* k) {
    const volatile kmd_args_t* a = k->args;
    if (events_open(k) != KMD_ERROR_NONE) {
        result_error(k, KMD_ERROR_WINDOW, 0, 0, 0, 0);
        return k->error;
    }
    events_word(k, KMD_HOST_STATE, KMD_STATE_BOOT);
    const uint64_t by = k->t_boot + a->t_start;
    volatile const uint32_t* mb = (volatile const uint32_t*)(k->lim + KMD_LIM_MAILBOX);
    result_where(k, KMD_WHERE_HELLO);
    kmd_wait_t w = wait_word(k, mb, 1u, by);
    if (w != KMD_OK) {
        result_error(
            k, w == KMD_STOP ? KMD_ERROR_STOPPED : KMD_ERROR_START, 0, 0, 0, 0
        );
        return k->error;
    }
    x280_fence_ir();
    const uint32_t control = mb[1];
    result_word(k, KMD_R_CONTROL, control);
    if (
        (control & 63u) || control < KMD_CONTROL_FLOOR ||
        control > KMD_L1_BYTES - KM_CONTROL_BYTES
    ) {
        result_error(k, KMD_ERROR_ALIGN, 0, 0, 0, control);
        return k->error;
    }
    if (grid_open(k, control) != KMD_ERROR_NONE) {
        result_error(k, KMD_ERROR_WINDOW, 0, 0, 0, 0);
        return k->error;
    }
    result_where(k, KMD_WHERE_READY);
    for (uint32_t c = 0; c < (uint32_t)a->columns; c++) {
        w = (
            wait_word(
                k,
                (volatile const uint32_t*)(k->lim + KMD_LIM_READY + c * 16u),
                KM_READY,
                by
            )
        );
        if (w != KMD_OK) {
            const uint32_t e = w == KMD_STOP ? KMD_ERROR_STOPPED : KMD_ERROR_START;
            result_error(k, e, 0, 0, 0, c);
            return k->error;
        }
    }
    return KMD_ERROR_NONE;
}

#ifdef KMD_VERIFY_LINES
/* Test build: checks each word of the user command in the host's line against its
 * command number. */
static void verify_line(kmd_t* k) {
    const kmd_ring_t* r = &k->ring[KMD_SRC_HOST];
    const uint64_t* l = r->line;
    const uint32_t i = r->next, hdr = (uint32_t)(l[0] >> 32);
    if ((hdr & 0xFFu) > KM_HANDLER_USER_MAX) {
        return;
    }
    const uint32_t w[12] = {
        hdr & 0xFFFFu,
        (uint32_t)l[1],
        (uint32_t)(l[1] >> 32),
        (uint32_t)(l[2] >> 32),
        (uint32_t)l[3],
        (uint32_t)(l[3] >> 32),
        (uint32_t)(l[4] >> 32),
        (uint32_t)l[5],
        (uint32_t)(l[5] >> 32),
        (uint32_t)(l[6] >> 32),
        (uint32_t)l[7],
        (uint32_t)(l[7] >> 32)
    };
    uint32_t bad = w[0] != 0 || w[1] != kmd_verify_group(i, k->cores);
    for (uint32_t j = 0; j < 10; j++) {
        bad |= w[2 + j] != kmd_verify_word(i, j);
    }
    k->checked++;
    if (bad && k->bad_lines++ == 0) {
        k->bad_first = i;
    }
}

/* Test build: KMD_HANDLER_CHECK_UPLOAD. Answers PONG with auxiliary = words that differ
 * (at most 0xFFFFFF) and ticks = the first one's index from `off` (~0: none). */
static void upload_check(
    kmd_t* k, uint32_t req, uint32_t off, uint32_t bytes, uint32_t seed
) {
    if (
        ((off | bytes) & 63u) || off > k->upload_bytes || bytes > k->upload_bytes - off
    ) {
        host_push(k, req, KM_EVENT_REJECTED | (KMD_REJECT_REGION << 8), 0);
        k->rejected++;
        return;
    }
    const volatile uint64_t* p = (const volatile uint64_t*)(k->upload + off);
    uint32_t bad = 0, first = ~0u;
    x280_fence_ir();
    for (uint32_t i = 0; i < bytes / 8u; i++) {
        const uint64_t v = p[i];
        const uint32_t j = off / 4u + 2u * i;
        const uint32_t lo = (uint32_t)v != kmd_verify_word(seed, j);
        const uint32_t hi = (uint32_t)(v >> 32) != kmd_verify_word(seed, j + 1u);
        if ((lo | hi) && first == ~0u) {
            first = 2u * i + (lo ? 0u : 1u);
        }
        bad += lo + hi;
    }
    host_push(
        k, req, KM_EVENT_PONG | ((bad < 0xFFFFFFu ? bad : 0xFFFFFFu) << 8), first
    );
}
#endif

/* Reads source `src`'s line: kernel commands act, bad ones are rejected, a command is
 * submitted or waits. STOP, EXIT, RUN, CANCEL: host only; KMD_HANDLER_POST: engine. */
KMD_HOT kmd_take_t take(kmd_t* k, uint32_t src, uint64_t now) {
    kmd_ring_t* r = &k->ring[src];
    uint64_t* l = r->line;
    const uint32_t req = r->next, host = src == KMD_SRC_HOST;
    uint32_t hdr = (uint32_t)(l[0] >> 32);
    const uint32_t handler = hdr & 0xFFu, flags = (hdr >> 8) & 0xFFu;
    const uint32_t group = (uint32_t)l[1], a0 = (uint32_t)(l[1] >> 32);
    const uint32_t first = group & 0xFFFFu, count = group >> 16;
    const uint32_t queued = handler != KM_HANDLER_EXIT && (flags & KM_FLAG_QUEUE);
    if (handler == KM_HANDLER_DEADLINE && a0 != 0) {
        ring_consumed(r);
        r->t_command = (uint64_t)a0 * 50u;
        return KMD_TAKEN;
    }
    if (handler == KM_HANDLER_STOP && host) {
        ring_consumed(r);
        if (k->busy) {
            result_error(k, KMD_ERROR_STOP_BUSY, 0, k->sequence, req, k->busy);
        }
        return KMD_FINISH;
    }
    if (handler == KMD_HANDLER_POST && src == KMD_SRC_ENGINE) {
        if (engine_post(k, l, 0)) {
            return KMD_NO_ROOM;
        }
        ring_consumed(r);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
    if (handler == KMD_HANDLER_DROP && src == KMD_SRC_ENGINE) {
        ring_consumed(r);
        queue_drop(k, a0);
        return KMD_TAKEN;
    }
    /* Every other command ends in one event: it waits until that event has room. */
    if (!ring_room(r)) {
        return KMD_NO_ROOM;
    }
    if (handler == KM_HANDLER_PING) {
        ring_consumed(r);
        ring_push(r, req, KM_EVENT_PONG, 0);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
    if (host && (handler == KM_HANDLER_RUN || handler == KM_HANDLER_CANCEL)) {
        /* a2 != 0: a graph's RUN; a CANCEL goes to the graph or stream with its req. */
        int wait = handler == KM_HANDLER_RUN ? -1 : graph_cancel(k, req, a0);
        if (handler == KM_HANDLER_RUN && (uint32_t)l[3] != 0) {
            wait = graph_run(k, req, l);
        } else if (wait < 0) {
            wait = (
                handler == KM_HANDLER_RUN
                    ? engine_run(k, req, a0, (uint32_t)(l[2] >> 32))
                    : engine_cancel(k, req, a0)
            );
        }
        if (wait) {
            return KMD_NO_ROOM;
        }
        ring_consumed(r);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
#ifdef KMD_VERIFY_LINES
    if (host && handler == KMD_HANDLER_CHECK_UPLOAD) {
        ring_consumed(r);
        upload_check(k, req, a0, (uint32_t)(l[2] >> 32), (uint32_t)l[3]);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
    if (host && handler == KMD_HANDLER_TRAP) {
        ring_consumed(r);
        const uint64_t v = *(volatile const uint64_t*)(uintptr_t)a0;
        ring_push(
            r, req, KM_EVENT_PONG, (uint32_t)v
        ); /* only if the load did not trap */
        return KMD_TAKEN;
    }
#endif
    uint32_t reject = 0;
    if (handler == KM_HANDLER_DEADLINE) {
        reject = KMD_REJECT_DEADLINE;
    } else if (
        (handler > KM_HANDLER_USER_MAX && handler != KM_HANDLER_EXIT) ||
        (!host && handler == KM_HANDLER_EXIT)
    ) {
        reject = KMD_REJECT_HANDLER;
    } else if (flags & 1u) {
        reject = KMD_REJECT_FLAGS;
    } else if (k->exited) {
        reject = KMD_REJECT_EXITED;
    } else if (
        handler != KM_HANDLER_EXIT && group_outside(k, queued ? 0 : first, count)
    ) {
        reject = KMD_REJECT_GROUP;
    } else if (queued && k->messages.handler != 0 && handler == k->messages.handler) {
        reject = KMD_REJECT_MESSAGES;
    }
    if (reject != 0) {
        ring_consumed(r);
        ring_push(r, req, KM_EVENT_REJECTED | (reject << 8), 0);
        k->t_last_event = now;
        k->rejected++;
        return KMD_TAKEN;
    }
    if (queued) {
        /* Into the queue; when full, the host's is rejected, others' rejected busy. */
        if (k->queue->n == KM_QUEUE_JOBS) {
            if (!host) {
                return KMD_HELD;
            }
            ring_consumed(r);
            ring_push(r, req, KM_EVENT_REJECTED | (KMD_REJECT_QUEUE << 8), 0);
            k->t_last_event = now;
            k->rejected++;
            return KMD_TAKEN;
        }
        queue_put(k, l, src, req, count, host ? 0u : hdr >> 16);
        ring_consumed(r);
        return KMD_TAKEN;
    }
    if (handler == KM_HANDLER_EXIT) {
        /* EXIT goes to every core, once no command is in flight or queued and no stream
         * runs. */
        if (k->busy || k->nstreams || k->queued || graph_running(k)) {
            return KMD_HELD;
        }
        l[1] = (l[1] & ~(uint64_t)0xFFFFFFFFu) | KM_GROUP_ALL(k->cores);
    } else if (
        (
            (flags & KM_FLAG_EXCLUSIVE) && (r->before || slots_overlap(k, first, count))
        ) ||
        messages_held(k, handler)
    ) {
        return KMD_HELD; /* exclusive: every job queued before it placed, no overlap */
    }
    const uint32_t slot = slot_free(k);
    if (slot == KM_RING_SLOTS) {
        return KMD_HELD; /* another source took the last free slot */
    }
    l[0] = (l[0] & 0xFFFFFFFFu) | (uint64_t)hdr << 32;
    command_submit(k, l, slot, req, src, now, r->t_command, KMD_TRACE_SUBMIT);
    messages_submitted(k, l, slot, req, src);
    ring_consumed(r);
    k->exited = handler == KM_HANDLER_EXIT;
    return KMD_TAKEN;
}

/* A command of the engine or a work cluster that cannot go now (no free slot, an
 * overlapping exclusive group, a full queue): rejected at once, KMD_EVENT_WAKE owed. */
static int source_busy(kmd_t* k, uint32_t src, uint64_t now) {
    kmd_ring_t* r = &k->ring[src];
    if (!(k->waits & (1u << src))) {
        r->owed++;
        if (!ring_room(r)) {
            r->owed--;
            return 0; /* no room for the rejection and the wake: it waits for room */
        }
        k->waits |= 1u << src;
    }
    const uint32_t hdr = (uint32_t)(r->line[0] >> 32);
    if (src == KMD_SRC_ENGINE && ((hdr >> 8) & KM_FLAG_QUEUE)) {
        /* A rejected job counts as taken in, as the engine counts every job sent. */
        *(volatile uint32_t*)(k->ering + KMD_ERING_TAKEN) = ++k->queue->taken;
    }
    ring_push(r, r->next, KM_EVENT_REJECTED | (KMD_REJECT_BUSY << 8), 0);
    x280_fence_ow(); /* the rejection lands before the consumed count passes it */
    ring_consumed(r);
    k->t_last_event = now;
    k->rejected++;
    return 1;
}

/* Source `src`'s turn in a pass: at most one command, the host's only while a slot is
 * free. Returns 1 when the command ends the loop. */
KMD_HOT int source_turn(kmd_t* k, uint32_t src, uint32_t* activity) {
    kmd_ring_t* r = &k->ring[src];
    const uint32_t host = src == KMD_SRC_HOST;
    if (r->line_state == 0) {
        if (k->busy == k->full && host) {
            k->full_passes += host;
            return 0;
        }
        if (!ring_take(r)) {
            k->idle_passes += host;
            return 0;
        }
        r->line_state = 1;
        r->before = k->queue->used;
        ++*activity;
#ifdef KMD_VERIFY_LINES
        if (host) {
            verify_line(k);
        }
#endif
    }
    const uint64_t now = x280_mtime();
    if (k->t_first == 0) {
        k->t_first = now;
    }
    const kmd_take_t t = take(k, src, now);
    if (t == KMD_FINISH) {
        return 1;
    }
    if (t == KMD_TAKEN) {
        r->line_state = 0;
    } else if (t == KMD_HELD && !host && source_busy(k, src, now)) {
        r->line_state = 0; /* only the host's commands wait */
    } else if (r->line_state == 1) {
        r->line_state = 2;
        if (t == KMD_HELD) {
            k->held++;
        } else {
            k->no_room++;
        }
    }
    return 0;
}

/* The loop: sweep, one command of each source in turn (host, engine), the queue's turn.
 * Returns when a STOP command, the runtime's stop or an error ends it. */
static void loop(kmd_t* k) {
    const volatile kmd_args_t* a = k->args;
    result_where(k, KMD_WHERE_LOOP);
    uint32_t spins = 0, activity = 0, seen_activity = 0;
    uint64_t t_active = x280_mtime();
    kmd_ring_t* host = &k->ring[KMD_SRC_HOST];
    kmd_ring_t* engine = &k->ring[KMD_SRC_ENGINE];
    for (;;) {
        k->passes++;
        if (k->busy) {
            slots_sweep(k);
        }
        if (k->messages.active) {
            messages_turn(k);
        }
        if (source_turn(k, KMD_SRC_HOST, &activity)) {
            return;
        }
        if (k->engine_on) {
            source_turn(k, KMD_SRC_ENGINE, &activity);
            ring_flush(engine);
            k->engine_on = engine_active(k);
        }
        if (k->queued != 0) {
            queue_turn(k, x280_mtime());
        }
        /* Work ring w is looked at on passes w mod 4, at most one ring a pass: one
         * look costs the idle pass 31 ns. */
        const uint32_t w = (uint32_t)(k->passes & 3u);
        if (w < KMD_WORK_CLUSTERS && k->ring[KMD_SRC_WORK + w].cmds != 0) {
            source_turn(k, KMD_SRC_WORK + w, &activity);
            ring_flush(&k->ring[KMD_SRC_WORK + w]);
        }
        ring_flush(host);
        if (++spins != KMD_POLL_EVERY) {
            continue;
        }
        spins = 0;
        k->x280_cores[KMD_H_MARKER0] = (uint32_t)(k->passes >> 10);
        if (x280_hart_stop_requested(k->core, k->run)) {
            result_error(k, KMD_ERROR_STOPPED, 0, k->sequence, 0, k->busy);
            return;
        }
        const uint64_t now = x280_mtime();
        const uint32_t late = slots_late(k, now);
        if (late != KM_RING_SLOTS) {
            /* The command's source gets the timeout event; the engine turns it into its
             * stream's error, which engine_stop forwards. */
            const uint32_t took = (uint32_t)(now - k->slot_t0[late]);
            const uint32_t src = k->slot_src[late];
            result_error(
                k,
                KMD_ERROR_DEADLINE,
                late,
                k->slot_sequence[late],
                k->slot_req[late],
                took
            );
            k->x280_cores[KMD_H_SOURCE] = src;
            const uint32_t auxiliary = (
                late |
                (k->slot_sequence[late] & KM_EVENT_SEQUENCE_MASK)
                    << KM_EVENT_SEQUENCE_SHIFT
            );
            ring_push(
                &k->ring[src],
                k->slot_req[late],
                KM_EVENT_TIMEOUT | auxiliary << 8,
                took
            );
            return;
        }
        if (
            activity != seen_activity || k->busy || k->nstreams || k->queued ||
            graph_running(k)
        ) {
            seen_activity = activity;
            t_active = now;
        } else if (a->t_idle != 0 && now - t_active > a->t_idle) {
            result_error(k, KMD_ERROR_IDLE, 0, k->sequence, 0, 0);
            return;
        }
    }
}

/* The stop flag goes out only while a slot is busy or before EXIT was submitted: after
 * a completed EXIT the cores have left and their L1 is no longer the program's. */
static void finish(kmd_t* k) {
    result_where(k, KMD_WHERE_FINISH);
    uint32_t abandoned = k->queued; /* no DONE event for these: dropped or cut off */
    for (uint32_t j = 0; j < KM_RING_SLOTS; j++) {
        abandoned += (k->busy >> j) & 1u;
    }
    k->x280_cores[KMD_H_ABANDONED] = abandoned;
    if (k->control[0] != 0 && (k->busy || !k->exited)) {
        grid_stop(k);
    }
    engine_stop(k);
    work_close(k);
    const uint64_t now = x280_mtime();
    result_finish(k, now);
#ifdef KMD_TRACE
    const uint32_t head[4] = {KMD_TRACE_MARK, k->traced, k->untraced, 0};
    if (k->upload_bytes >= 64u) {
        kmd_store_chunk((volatile uint8_t*)k->upload, head);
    }
#endif
    ring_flush(&k->ring[KMD_SRC_HOST]);
    events_word(k, KMD_HOST_ERROR, k->error);
    events_word(k, KMD_HOST_STATE, k->error ? KMD_STATE_FAILED : KMD_STATE_DONE);
    grid_close(k);
    events_close(k);
    const uint64_t ticks = k->t_first ? now - k->t_first : 0;
    const uint64_t core = k->core;
    const uint64_t result = k->error ? ~(uint64_t)k->error : ticks;
    kmd_flush_lines(k, KMD_STATE_BYTES);
    x280_hart_set_result(core, result);
}

/* After an exception on core `core` (trap.h). Core 0 finishes as an abort does from LIM
 * and the argument block alone: the state's pointers may be what failed. */
__attribute__((used, noinline)) void kmd_trapped(
    uint64_t core, uint64_t cause, uint64_t epc, uint64_t tval
) {
    const uint64_t run_value = x280_hart_control(core)->run_value;
    const volatile kmd_args_t* a = (const volatile kmd_args_t*)x280_hart_args(core);
    x280_hart_set_result(core, ~(uint64_t)KMD_ERROR_TRAP);
    if (a == 0 || !args_valid(a, run_value) || core > 1) {
        return;
    }
    volatile uint8_t* lim = (volatile uint8_t*)a;
    volatile uint32_t* x280_cores = (volatile uint32_t*)(lim + KMD_LIM_CORES);
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
    kmd_t* t = &fresh;
    t->core = core;
    t->run = run_value;
    t->args = a;
    t->lim = lim;
    t->result = (volatile uint32_t*)(lim + KMD_LIM_RESULT);
    t->x280_cores = x280_cores;
    t->panels = (uint32_t)a->panels;
    const uint32_t where = t->result[KMD_R_WHERE];
    result_error(t, KMD_ERROR_TRAP, (uint32_t)cause, (uint32_t)epc, 0, (uint32_t)tval);
    /* The old state's words, read but never followed: whether EXIT left the grid. */
    volatile uint8_t* state = (
        (volatile uint8_t*)(X280_LOCAL_DDR_TILE_BASE + a->state_ddr)
    );
    const volatile kmd_t* old = (const volatile kmd_t*)state;
    const uint32_t control = ((volatile const uint32_t*)(lim + KMD_LIM_MAILBOX))[1];
    if (
        where >= KMD_WHERE_READY && (old->busy || !old->exited) &&
        grid_open(t, control) == KMD_ERROR_NONE
    ) {
        grid_stop(t);
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
    const uint64_t by = x280_mtime() + KMD_ENGINE_STOP_TICKS;
    while (
        (
            x280_cores[KMD_H_ENGINE] == KMD_ENGINE_WAITING ||
            x280_cores[KMD_H_ENGINE] == KMD_ENGINE_RUNNING
        ) &&
        x280_mtime() <= by
    ) {
        x280_fence_ir();
    }
    if (where >= KMD_WHERE_HELLO && events_open(t) == KMD_ERROR_NONE) {
        events_word(t, KMD_HOST_ERROR, KMD_ERROR_TRAP);
        events_word(t, KMD_HOST_STATE, KMD_STATE_FAILED);
    }
    grid_close(t);
    events_close(t);
    for (uint32_t w = 0; w < KMD_WORK_CLUSTERS; w++) {
        noc_clear_tlb_2m(KMD_WIN_WORK + w);
    }
    kmd_flush_lines(state, KMD_STATE_BYTES);
    x280_fence_ow();
    result_word(t, KMD_R_STATE, KMD_STATE_FAILED);
}

/* Core 0's kernel; kept out of main so main's prologue touches no FP register. */
static __attribute__((noinline)) void run(uint64_t core) {
    uint64_t host_line[16], engine_line[16], work_lines[8u * KMD_WORK_CLUSTERS + 8u];
    const uint64_t run_value = x280_hart_control(core)->run_value;
    const uint64_t t_boot = x280_mtime();
    const volatile kmd_args_t* a = (const volatile kmd_args_t*)x280_hart_args(core);
    if (a == 0 || !args_valid(a, run_value)) {
        x280_hart_set_result(core, ~(uint64_t)KMD_ERROR_ARGS);
        return;
    }
    kmd_t* k = (kmd_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->state_ddr);
    /* The whole state area, engine's too, zeroed through a volatile pointer: a plain
     * loop may become a memset call, and the image has no C library. */
    volatile uint64_t* raw = (volatile uint64_t*)k;
    for (uint32_t i = 0; i < KMD_STATE_BYTES / sizeof(uint64_t); i++) {
        raw[i] = 0;
    }
    k->core = core;
    k->run = run_value;
    k->t_boot = t_boot;
    k->args = a;
    k->lim = (volatile uint8_t*)a;
    k->result = (volatile uint32_t*)(k->lim + KMD_LIM_RESULT);
    k->x280_cores = (volatile uint32_t*)(k->lim + KMD_LIM_CORES);
    k->upload = k->lim + KMD_LIM_UPLOAD;
    k->upload_bytes = (uint32_t)a->upload_bytes;
    k->panels = (uint32_t)a->panels;
    k->depth = (uint32_t)a->depth;
    k->full = (1u << k->depth) - 1u;
    k->cores = (uint32_t)a->cores;
    k->messages.handler = (uint32_t)a->messages_handler;
    k->ering = (volatile uint8_t*)k + KMD_STATE_RING;
    k->queue = (kmd_queue_t*)((uint8_t*)k + KMD_STATE_QUEUE);
    kmd_ring_t* h = &k->ring[KMD_SRC_HOST];
    h->cmds = k->lim + KMD_LIM_HRING;
    h->read = (volatile const uint32_t*)(k->lim + KMD_LIM_READ);
    h->depth = (uint32_t)a->host_depth;
    h->event_depth = (uint32_t)a->event_depth;
    h->next = 1;
    h->t_command = a->t_command;
    h->line = (uint64_t*)(((uintptr_t)host_line + 63u) & ~(uintptr_t)63u);
    result_word(k, KMD_R_STATE, KMD_STATE_BOOT);
    uint64_t* wl = (uint64_t*)(((uintptr_t)work_lines + 63u) & ~(uintptr_t)63u);
    if (boot(k) == KMD_ERROR_NONE && work_open(k, wl) != KMD_ERROR_NONE) {
        result_error(k, KMD_ERROR_WINDOW, 0, 0, 0, 0);
    }
    if (k->error == KMD_ERROR_NONE) {
        engine_open(k, (uint64_t*)(((uintptr_t)engine_line + 63u) & ~(uintptr_t)63u));
        engine_go(k, 1u);
        result_word(k, KMD_R_BOOT_TICKS, (uint32_t)(x280_mtime() - k->t_boot));
        result_word(k, KMD_R_STATE, KMD_STATE_READY);
        events_ready(k);
        events_word(k, KMD_HOST_STATE, KMD_STATE_READY);
        loop(k);
    } else {
        engine_go(k, 2u);
    }
    finish(k);
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
        const volatile kmd_args_t* a = (
            (const volatile kmd_args_t*) x280_hart_args(core_id)
        );
        if (a != 0 && args_valid(a, run_value)) {
            eng_run(core_id, a, run_value);
        }
    }
    return 0;
}
