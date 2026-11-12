/* The dispatch kernel on X280 cluster 0, core 0 (cores 1 to 3 return at once): boot,
 * then the loop over ack words and host commands, then finish (stop word, result). */
#include <stdint.h>

#include "completion.h"
#include "event_ring.h"
#include "grid_port.h"
#include "host_ring.h"
#include "kernel.h"
#include "result.h"

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

/* Checks the argument block against this run. */
static int args_valid(const volatile kmd_args_t* a, uint64_t run) {
    if (x280_hart_args_bytes() < sizeof(kmd_args_t) || a->run_value != run) {
        return 0;
    }
    const uint64_t host_bytes = KMD_HOST_RING + a->event_depth * sizeof(km_event_t);
    return (
        a->depth >= 1 && a->depth <= KM_RING_SLOTS && a->host_depth >= 1 &&
        a->host_depth <= KMD_HOST_DEPTH_MAX && a->event_depth >= 1 && a->columns >= 1 &&
        a->columns <= KMD_READY_MAX && a->cores >= 1 && a->cores <= KM_CORES_MAX &&
        a->panels >= 1 && a->panels <= 2 && a->t_record != 0 && a->state_ddr != 0 &&
        (a->state_ddr & 63u) == 0 &&
        (a->host_noc_addr & (NOC_2M_WINDOW_STRIDE - 1u)) + host_bytes <=
            NOC_2M_WINDOW_STRIDE
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

/* Takes command k->next from k->line. Kernel commands act here; a record gets the
 * lowest free slot and the next sequence and goes to the grid. Returns 1 to finish. */
static int take(kmd_t* k, uint64_t now) {
    uint64_t* l = k->line;
    const uint32_t req = k->next, hdr = (uint32_t)(l[0] >> 32);
    const uint32_t seg = hdr & 0xFFu, flags = (hdr >> 8) & 0xFFu;
    const uint32_t team = (uint32_t)l[1], a0 = (uint32_t)(l[1] >> 32);
    const uint32_t first = team & 0xFFFFu, count = team >> 16;
    if (seg == KM_SEG_PING) {
        hring_consumed(k);
        events_push(k, req, KM_EV_PONG, 0);
        k->t_last_event = now;
        return 0;
    }
    if (seg == KM_SEG_DEADLINE) {
        hring_consumed(k);
        k->t_record = a0 ? (uint64_t)a0 * 50u : k->args->t_record;
        return 0;
    }
    if (seg == KM_SEG_STOP) {
        hring_consumed(k);
        if (k->busy) {
            result_error(k, KMD_ERR_STOP_BUSY, 0, k->seq, req, k->busy);
        }
        return 1;
    }
    if ((seg > KM_SEG_USER_MAX && seg != KM_SEG_EXIT) || (flags & 1u) || k->exited ||
        count == 0 || first + count > k->cores) {
        result_error(k, KMD_ERR_COMMAND, 0, k->seq + 1u, req, hdr);
        return 1;
    }
    const uint32_t slot = slot_free(k), seq = ++k->seq;
    l[0] = seq | ((uint64_t)((hdr & 0xFFFFu) | (slot << 16)) << 32);
    for (uint32_t j = 2; j < 8; j += 2) {
        l[j] = (l[j] & ~(uint64_t)0xFFFFFFFFu) | seq;
    }
    grid_store_record(k, slot);
    hring_consumed(k);
    k->records++;
    k->exited = seg == KM_SEG_EXIT;
    slot_take(k, slot, seq, req, now, k->t_record);
    return 0;
}

/* The loop. Returns when a STOP command, the runtime's stop or an error ends it. */
static void loop(kmd_t* k) {
    const volatile kmd_args_t* a = k->args;
    result_where(k, KMD_WHERE_LOOP);
    uint32_t spins = 0, activity = 0, seen_activity = 0;
    uint64_t t_active = x280_mtime();
    for (;;) {
        k->passes++;
        if (k->busy) {
            slots_sweep(k);
        }
        if (k->busy == k->full) {
            k->full_passes++;
        } else if (!hring_take(k)) {
            k->idle_passes++;
        } else {
            activity++;
            const uint64_t now = x280_mtime();
            if (k->t_first == 0) {
                k->t_first = now;
            }
            if (take(k, now)) {
                return;
            }
        }
        events_flush(k);
        if (++spins != KMD_POLL_EVERY) {
            continue;
        }
        spins = 0;
        if (x280_hart_stop_requested(k->hart, k->run)) {
            result_error(k, KMD_ERR_STOPPED, 0, k->seq, 0, k->busy);
            return;
        }
        const uint64_t now = x280_mtime();
        const uint32_t late = slots_late(k, now);
        if (late != KM_RING_SLOTS) {
            result_error(
                k,
                KMD_ERR_DEADLINE,
                late,
                k->slot_seq[late],
                k->slot_req[late],
                (uint32_t)(now - k->slot_t0[late])
            );
            events_push(
                k,
                k->slot_req[late],
                KM_EV_TIMEOUT | (late << 8),
                (uint32_t)(now - k->slot_t0[late])
            );
            return;
        }
        if (activity != seen_activity || k->busy) {
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
    const uint64_t now = x280_mtime();
    result_finish(k, now);
    events_flush(k);
    events_word(k, KMD_HOST_ERROR, k->error);
    events_word(k, KMD_HOST_STATE, k->error ? KMD_STATE_FAILED : KMD_STATE_DONE);
    grid_close(k);
    events_close(k);
    const uint64_t ticks = k->t_first ? now - k->t_first : 0;
    x280_hart_set_result(k->hart, k->error ? ~(uint64_t)k->error : ticks);
}

/* Core 0's kernel; kept out of main so main's prologue touches no FP register. */
static __attribute__((noinline)) void run(uint64_t hart) {
    uint64_t line_buf[24];
    const uint64_t run_value = x280_hart_control(hart)->run_value;
    const uint64_t t_boot = x280_mtime();
    const volatile kmd_args_t* a = (const volatile kmd_args_t*)x280_hart_args(hart);
    if (a == 0 || !args_valid(a, run_value)) {
        x280_hart_set_result(hart, ~(uint64_t)KMD_ERR_ARGS);
        return;
    }
    kmd_t* k = (kmd_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->state_ddr);
    /* Zeroed through a volatile pointer: a plain loop may become a memset call, and the
     * image links without a C library. */
    volatile uint64_t* raw = (volatile uint64_t*)k;
    for (uint32_t i = 0; i < sizeof(*k) / sizeof(uint64_t); i++) {
        raw[i] = 0;
    }
    k->hart = hart;
    k->line = (uint64_t*)(((uintptr_t)line_buf + 63u) & ~(uintptr_t)63u);
    k->run = run_value;
    k->t_boot = t_boot;
    k->args = a;
    k->lim = (volatile uint8_t*)a;
    k->result = (volatile uint32_t*)(k->lim + KMD_LIM_RESULT);
    k->panels = (uint32_t)a->panels;
    k->host_depth = (uint32_t)a->host_depth;
    k->event_depth = (uint32_t)a->event_depth;
    k->depth = (uint32_t)a->depth;
    k->full = (1u << k->depth) - 1u;
    k->cores = (uint32_t)a->cores;
    k->t_record = a->t_record;
    k->next = 1;
    result_word(k, KMD_R_STATE, KMD_STATE_BOOT);
    if (boot(k) == KMD_ERR_NONE) {
        result_word(k, KMD_R_BOOT_TICKS, (uint32_t)(x280_mtime() - k->t_boot));
        result_word(k, KMD_R_STATE, KMD_STATE_READY);
        events_word(k, KMD_HOST_STATE, KMD_STATE_READY);
        loop(k);
    }
    finish(k);
}

int main(uint64_t hartid) {
    /* Floating point and vector on first, on every hart: the harness leaves them off
     * and the first such instruction would trap. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    if (hartid == 0) {
        run(hartid);
    }
    return 0;
}
