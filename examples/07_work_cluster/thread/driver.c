/* The control loop on the work cluster: decide.c's commands through the ring until
 * SUM's result word ends it. Codes, words and region: see README.md. */
#include <stdint.h>

#include "decide.h"
#include "kmx.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

/* Fills r field by field (a struct assignment may call memset, which does not link). */
static void command(const wc_command* w, km_command_t* r) {
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = w->handler | w->flags << 8;
    r->group = w->first | w->count << 16;
    r->a0 = w->a0;
    r->a1 = w->a1;
    r->a2 = r->a3 = r->a4 = r->a5 = r->a6 = r->a7 = r->a8 = r->a9 = 0;
}

/* Submits r, waiting while the ring is full; KMX_OK and its handle, or the error. */
static int submit(const km_command_t* r, uint32_t* handle) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    int e = kmx_submit(r, handle);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_submit(r, handle);
    }
    return e;
}

static void finish(uint32_t code, const wc_state* s) {
    kmx_flush(0);
    const uint32_t words[3] = {s->round, s->sum, s->trail};
    kmx_result(code, words, 3);
}

int kmx_main(int core) {
    (void)core;
    const kmx_args* a = kmx_args_get();
    wc_state s;
    uint32_t words[WC_WORDS];
    for (int i = 0; i < WC_WORDS; i++) {
        words[i] = i < (int)a->nwords ? a->words[i] : 0;
    }
    uint64_t bytes;
    volatile uint32_t* groups = (volatile uint32_t*)kmx_region(0, &bytes);
    if (
        a->nwords < WC_WORDS || wc_start(&s, words) || groups == 0 ||
        bytes < (uint64_t)words[WC_MAX_ROUNDS] * 4
    ) {
        return 1;
    }
    for (;;) {
        wc_command w[2];
        km_command_t r;
        uint32_t handle[2];
        kmx_event ev[2];
        wc_round(&s, w);
        groups[s.round] = w[0].first;
        kmx_progress(s.round * 4 + 1);
        /* One command at a time: the kernel rejects a thread's command that would wait
         * (SUM is exclusive), so SUM goes once STEP's event is in. */
        for (int i = 0; i < 2; i++) {
            command(&w[i], &r);
            if (submit(&r, &handle[i]) != KMX_OK) {
                finish(3, &s);
                return 0;
            }
            kmx_progress(s.round * 4 + 2);
            const int e = kmx_wait(handle[i], kmx_time() + WAIT_TICKS, &ev[i]);
            if (e != KMX_OK) {
                finish(3, &s);
                return 0;
            }
            if (ev[i].kind != KM_EV_DONE) {
                finish(4, &s);
                return 0;
            }
        }
        kmx_progress(s.round * 4 + 3);
        const uint32_t next = wc_result(&s, ev[1].ticks);
        if (next != WC_AGAIN) {
            finish(next == WC_END ? 0u : 2u, &s);
            return 0;
        }
    }
}
