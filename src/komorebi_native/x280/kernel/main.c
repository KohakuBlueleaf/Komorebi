/* Dispatch kernel, X280 cluster 0: core 0 boots, loops over acks and the sources'
 * commands, finishes; core 1 runs the stream engine; cores 2 and 3 return. */
#include <stdint.h>

#include "completion.h"
#include "engine.h"
#include "engine_port.h"
#include "event_ring.h"
#include "graph_port.h"
#include "grid_port.h"
#include "kernel.h"
#include "queue.h"
#include "result.h"
#include "ring.h"
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
    if (x280_hart_args_bytes() < sizeof(kmd_args_t) || a->run_value != run) {
        return 0;
    }
    const uint64_t host_bytes = KMD_HOST_RING + a->event_depth * sizeof(km_event_t);
    return (
        a->depth >= 1 && a->depth <= KM_RING_SLOTS && pow2(a->host_depth) &&
        a->host_depth <= KMD_HOST_DEPTH_MAX && pow2(a->event_depth) &&
        a->columns >= 1 && a->columns <= KMD_READY_MAX && a->cores >= 1 &&
        a->cores <= KM_CORES_MAX && a->panels >= 1 && a->panels <= 2 &&
        a->t_start != 0 && a->t_record != 0 && a->state_ddr != 0 &&
        (a->upload_bytes & 63u) == 0 && a->upload_bytes <= X280_LIM_SIZE &&
        (uintptr_t)a + KMD_LIM_UPLOAD + a->upload_bytes <= X280_LIM_END &&
        (a->state_ddr & 63u) == 0 && (a->host_noc_addr & 63u) == 0 &&
        (a->host_noc_addr & (NOC_2M_WINDOW_STRIDE - 1u)) + host_bytes <=
            NOC_2M_WINDOW_STRIDE &&
        work_args_valid(a)
    );
}

/* Start-up: host window, hello, panels (so a failed READY wait can stop the grid),
 * READY. Returns KMD_ERR_NONE or an error. */
static uint32_t boot(kmd_t* k) {
    const volatile kmd_args_t* a = k->args;
    if (events_open(k) != KMD_ERR_NONE) {
        result_error(k, KMD_ERR_WINDOW, 0, 0, 0, 0);
        return k->error;
    }
    events_word(k, KMD_HOST_STATE, KMD_STATE_BOOT);
    const uint64_t by = k->t_boot + a->t_start;
    volatile const uint32_t* mb = (volatile const uint32_t*)(k->lim + KMD_LIM_MAILBOX);
    result_where(k, KMD_WHERE_HELLO);
    kmd_wait_t w = wait_word(k, mb, 1u, by);
    if (w != KMD_OK) {
        result_error(k, w == KMD_STOP ? KMD_ERR_STOPPED : KMD_ERR_START, 0, 0, 0, 0);
        return k->error;
    }
    x280_fence_ir();
    const uint32_t ctl = mb[1];
    result_word(k, KMD_R_CTL, ctl);
    if ((ctl & 63u) || ctl < KMD_CTL_FLOOR || ctl > KMD_L1_BYTES - KM_CTL_BYTES) {
        result_error(k, KMD_ERR_ALIGN, 0, 0, 0, ctl);
        return k->error;
    }
    if (grid_open(k, ctl) != KMD_ERR_NONE) {
        result_error(k, KMD_ERR_WINDOW, 0, 0, 0, 0);
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
            const uint32_t e = w == KMD_STOP ? KMD_ERR_STOPPED : KMD_ERR_START;
            result_error(k, e, 0, 0, 0, c);
            return k->error;
        }
    }
    return KMD_ERR_NONE;
}

#ifdef KMD_VERIFY_LINES
/* Test build: checks each word of the user record in the host's line against its
 * command number. */
static void verify_line(kmd_t* k) {
    const kmd_ring_t* r = &k->ring[KMD_SRC_HOST];
    const uint64_t* l = r->line;
    const uint32_t i = r->next, hdr = (uint32_t)(l[0] >> 32);
    if ((hdr & 0xFFu) > KM_SEG_USER_MAX) {
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
    uint32_t bad = w[0] != 0 || w[1] != kmd_verify_team(i, k->cores);
    for (uint32_t j = 0; j < 10; j++) {
        bad |= w[2 + j] != kmd_verify_word(i, j);
    }
    k->checked++;
    if (bad && k->bad_lines++ == 0) {
        k->bad_first = i;
    }
}

/* Test build: KMD_SEG_CHECK_UPLOAD. Answers PONG with aux = words that differ (at most
 * 0xFFFFFF) and ticks = the first one's index from `off` (~0: none). */
static void upload_check(
    kmd_t* k, uint32_t req, uint32_t off, uint32_t bytes, uint32_t seed
) {
    if (((off | bytes) & 63u) || off > k->upload_bytes ||
        bytes > k->upload_bytes - off) {
        host_push(k, req, KM_EV_REFUSED | (KMD_REFUSE_REGION << 8), 0);
        k->refused++;
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
    host_push(k, req, KM_EV_PONG | ((bad < 0xFFFFFFu ? bad : 0xFFFFFFu) << 8), first);
}
#endif

/* Takes source `src`'s line: kernel commands act, bad records are refused, a record is
 * issued or waits. STOP, EXIT, RUN, CANCEL: host only; KMD_SEG_POST: engine only. */
KMD_HOT kmd_take_t take(kmd_t* k, uint32_t src, uint64_t now) {
    kmd_ring_t* r = &k->ring[src];
    uint64_t* l = r->line;
    const uint32_t req = r->next, host = src == KMD_SRC_HOST;
    uint32_t hdr = (uint32_t)(l[0] >> 32);
    const uint32_t seg = hdr & 0xFFu, flags = (hdr >> 8) & 0xFFu;
    const uint32_t team = (uint32_t)l[1], a0 = (uint32_t)(l[1] >> 32);
    const uint32_t first = team & 0xFFFFu, count = team >> 16;
    const uint32_t queued = seg != KM_SEG_EXIT && (flags & KM_F_QUEUE);
    if (seg == KM_SEG_DEADLINE && a0 != 0) {
        ring_consumed(r);
        r->t_record = (uint64_t)a0 * 50u;
        return KMD_TAKEN;
    }
    if (seg == KM_SEG_STOP && host) {
        ring_consumed(r);
        if (k->busy) {
            result_error(k, KMD_ERR_STOP_BUSY, 0, k->seq, req, k->busy);
        }
        return KMD_FINISH;
    }
    if (seg == KMD_SEG_POST && src == KMD_SRC_ENGINE) {
        if (engine_post(k, l, 0)) {
            return KMD_NO_ROOM;
        }
        ring_consumed(r);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
    if (seg == KMD_SEG_DROP && src == KMD_SRC_ENGINE) {
        ring_consumed(r);
        queue_drop(k, a0);
        return KMD_TAKEN;
    }
    /* Every other command ends in one event: it waits until that event has room. */
    if (!ring_room(r)) {
        return KMD_NO_ROOM;
    }
    if (seg == KM_SEG_PING) {
        ring_consumed(r);
        ring_push(r, req, KM_EV_PONG, 0);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
    if (host && (seg == KM_SEG_RUN || seg == KM_SEG_CANCEL)) {
        /* a2 != 0: a graph's RUN; a CANCEL goes to the graph or stream with its req. */
        int wait = seg == KM_SEG_RUN ? -1 : graph_cancel(k, req, a0);
        if (seg == KM_SEG_RUN && (uint32_t)l[3] != 0) {
            wait = graph_run(k, req, l);
        } else if (wait < 0) {
            wait =
                (seg == KM_SEG_RUN ? engine_run(k, req, a0, (uint32_t)(l[2] >> 32))
                                   : engine_cancel(k, req, a0));
        }
        if (wait) {
            return KMD_NO_ROOM;
        }
        ring_consumed(r);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
#ifdef KMD_VERIFY_LINES
    if (host && seg == KMD_SEG_CHECK_UPLOAD) {
        ring_consumed(r);
        upload_check(k, req, a0, (uint32_t)(l[2] >> 32), (uint32_t)l[3]);
        k->t_last_event = now;
        return KMD_TAKEN;
    }
#endif
    uint32_t refuse = 0;
    if (seg == KM_SEG_DEADLINE) {
        refuse = KMD_REFUSE_DEADLINE;
    } else if (
        (seg > KM_SEG_USER_MAX && seg != KM_SEG_EXIT) || (!host && seg == KM_SEG_EXIT)
    ) {
        refuse = KMD_REFUSE_SEG;
    } else if (flags & 1u) {
        refuse = KMD_REFUSE_FLAGS;
    } else if (k->exited) {
        refuse = KMD_REFUSE_EXITED;
    } else if (seg != KM_SEG_EXIT && team_outside(k, queued ? 0 : first, count)) {
        refuse = KMD_REFUSE_TEAM;
    }
    if (refuse != 0) {
        ring_consumed(r);
        ring_push(r, req, KM_EV_REFUSED | (refuse << 8), 0);
        k->t_last_event = now;
        k->refused++;
        return KMD_TAKEN;
    }
    if (queued) {
        /* Into the queue; when full, the host's is refused, others' refused busy. */
        if (k->queue->n == KM_QUEUE_JOBS) {
            if (!host) {
                return KMD_HELD;
            }
            ring_consumed(r);
            ring_push(r, req, KM_EV_REFUSED | (KMD_REFUSE_QUEUE << 8), 0);
            k->t_last_event = now;
            k->refused++;
            return KMD_TAKEN;
        }
        queue_put(k, l, src, req, count, host ? 0u : hdr >> 16);
        ring_consumed(r);
        return KMD_TAKEN;
    }
    if (seg == KM_SEG_EXIT) {
        /* EXIT goes to every core, once no record is in flight or queued and no stream
         * runs. */
        if (k->busy || k->nstreams || k->queued || graph_running(k)) {
            return KMD_HELD;
        }
        l[1] = (l[1] & ~(uint64_t)0xFFFFFFFFu) | KM_TEAM_ALL(k->cores);
    } else if ((flags & KM_F_EXCL) && slots_overlap(k, first, count)) {
        return KMD_HELD;
    }
    const uint32_t slot = slot_free(k);
    if (slot == KM_RING_SLOTS) {
        return KMD_HELD; /* another source took the last free slot */
    }
    l[0] = (l[0] & 0xFFFFFFFFu) | (uint64_t)hdr << 32;
    record_issue(k, l, slot, req, src, now, r->t_record, KMD_TRACE_ISSUE);
    ring_consumed(r);
    k->exited = seg == KM_SEG_EXIT;
    return KMD_TAKEN;
}

/* Source `src`'s turn in a pass: at most one command, taken unless every slot is busy
 * (the engine's always). Returns 1 when the command ends the loop. */
KMD_HOT int source_turn(kmd_t* k, uint32_t src, uint32_t* activity) {
    kmd_ring_t* r = &k->ring[src];
    const uint32_t host = src == KMD_SRC_HOST;
    if (r->line_state == 0) {
        if (k->busy == k->full && src != KMD_SRC_ENGINE) {
            k->full_passes += host;
            return 0;
        }
        if (!ring_take(r)) {
            k->idle_passes += host;
            return 0;
        }
        r->line_state = 1;
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
    } else if (t == KMD_HELD && src == KMD_SRC_WORK) {
        work_busy(k, now); /* a thread's record never waits */
        r->line_state = 0;
    } else if (t == KMD_HELD && src == KMD_SRC_ENGINE && engine_busy(k, now)) {
        r->line_state = 0; /* nor does the engine's */
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
    kmd_ring_t* work = &k->ring[KMD_SRC_WORK];
    for (;;) {
        k->passes++;
        if (k->busy) {
            slots_sweep(k);
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
        /* Every fourth pass: one look at the work ring costs the idle pass 31 ns. */
        if (work->cmds != 0 && (k->passes & 3u) == 0) {
            source_turn(k, KMD_SRC_WORK, &activity);
            ring_flush(work);
        }
        ring_flush(host);
        if (++spins != KMD_POLL_EVERY) {
            continue;
        }
        spins = 0;
        k->harts[KMD_H_STAMP0] = (uint32_t)(k->passes >> 10);
        if (x280_hart_stop_requested(k->hart, k->run)) {
            result_error(k, KMD_ERR_STOPPED, 0, k->seq, 0, k->busy);
            return;
        }
        const uint64_t now = x280_mtime();
        const uint32_t late = slots_late(k, now);
        if (late != KM_RING_SLOTS) {
            /* The record's source gets the timeout event; the engine turns it into its
             * stream's error, which engine_stop forwards. */
            const uint32_t took = (uint32_t)(now - k->slot_t0[late]);
            const uint32_t src = k->slot_src[late];
            result_error(
                k, KMD_ERR_DEADLINE, late, k->slot_seq[late], k->slot_req[late], took
            );
            k->harts[KMD_H_SOURCE] = src;
            ring_push(
                &k->ring[src], k->slot_req[late], KM_EV_TIMEOUT | (late << 8), took
            );
            return;
        }
        if (activity != seen_activity || k->busy || k->nstreams || k->queued ||
            graph_running(k)) {
            seen_activity = activity;
            t_active = now;
        } else if (a->t_idle != 0 && now - t_active > a->t_idle) {
            result_error(k, KMD_ERR_IDLE, 0, k->seq, 0, 0);
            return;
        }
    }
}

/* The stop word goes out only while a slot is busy or before EXIT was issued: after a
 * completed EXIT the cores have left and their L1 is no longer the program's. */
static void finish(kmd_t* k) {
    result_where(k, KMD_WHERE_FINISH);
    if (k->ctl[0] != 0 && (k->busy || !k->exited)) {
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
    const uint64_t hart = k->hart;
    const uint64_t result = k->error ? ~(uint64_t)k->error : ticks;
    kmd_flush_lines(k, KMD_STATE_BYTES);
    x280_hart_set_result(hart, result);
}

/* Core 0's kernel; kept out of main so main's prologue touches no FP register. */
static __attribute__((noinline)) void run(uint64_t hart) {
    uint64_t host_line[16], engine_line[16], work_line[16];
    const uint64_t run_value = x280_hart_control(hart)->run_value;
    const uint64_t t_boot = x280_mtime();
    const volatile kmd_args_t* a = (const volatile kmd_args_t*)x280_hart_args(hart);
    if (a == 0 || !args_valid(a, run_value)) {
        x280_hart_set_result(hart, ~(uint64_t)KMD_ERR_ARGS);
        return;
    }
    kmd_t* k = (kmd_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->state_ddr);
    /* The whole state area, engine's too, zeroed through a volatile pointer: a plain
     * loop may become a memset call, and the image has no C library. */
    volatile uint64_t* raw = (volatile uint64_t*)k;
    for (uint32_t i = 0; i < KMD_STATE_BYTES / sizeof(uint64_t); i++) {
        raw[i] = 0;
    }
    k->hart = hart;
    k->run = run_value;
    k->t_boot = t_boot;
    k->args = a;
    k->lim = (volatile uint8_t*)a;
    k->result = (volatile uint32_t*)(k->lim + KMD_LIM_RESULT);
    k->harts = (volatile uint32_t*)(k->lim + KMD_LIM_HARTS);
    k->upload = k->lim + KMD_LIM_UPLOAD;
    k->upload_bytes = (uint32_t)a->upload_bytes;
    k->panels = (uint32_t)a->panels;
    k->depth = (uint32_t)a->depth;
    k->full = (1u << k->depth) - 1u;
    k->cores = (uint32_t)a->cores;
    k->ering = (volatile uint8_t*)k + KMD_STATE_RING;
    k->queue = (kmd_queue_t*)((uint8_t*)k + KMD_STATE_QUEUE);
    kmd_ring_t* h = &k->ring[KMD_SRC_HOST];
    h->cmds = k->lim + KMD_LIM_HRING;
    h->read = (volatile const uint32_t*)(k->lim + KMD_LIM_READ);
    h->depth = (uint32_t)a->host_depth;
    h->event_depth = (uint32_t)a->event_depth;
    h->next = 1;
    h->t_record = a->t_record;
    h->line = (uint64_t*)(((uintptr_t)host_line + 63u) & ~(uintptr_t)63u);
    result_word(k, KMD_R_STATE, KMD_STATE_BOOT);
    uint64_t* wl = (uint64_t*)(((uintptr_t)work_line + 63u) & ~(uintptr_t)63u);
    if (boot(k) == KMD_ERR_NONE && work_open(k, wl) != KMD_ERR_NONE) {
        result_error(k, KMD_ERR_WINDOW, 0, 0, 0, 0);
    }
    if (k->error == KMD_ERR_NONE) {
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

int main(uint64_t hartid) {
    /* Floating point and vector on first, on every hart: the harness leaves them off
     * and the first such instruction would trap. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    if (hartid == 0) {
        run(hartid);
    } else if (hartid == 1) {
        const uint64_t run_value = x280_hart_control(hartid)->run_value;
        const volatile kmd_args_t* a = (
            (const volatile kmd_args_t*) x280_hart_args(hartid)
        );
        if (a != 0 && args_valid(a, run_value)) {
            eng_run(hartid, a, run_value);
        }
    }
    return 0;
}
