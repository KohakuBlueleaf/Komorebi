/* The control loop on the work cluster, with the class's commands: decide.c's rounds
 * through the ring until SUM's result word ends it, each command's handler and words
 * placed by the class's generated header (workcluster_thread.h). See ../README.md. */
#include <stdint.h>

#include "decide.h"
#include "kmx.h"
#include "workcluster_thread.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */

/* Word `place` (a0 is 0) of `command`. */
static void put(km_command_t* command, uint32_t place, uint32_t value) {
    uint32_t* words[10] = {
        &command->a0,
        &command->a1,
        &command->a2,
        &command->a3,
        &command->a4,
        &command->a5,
        &command->a6,
        &command->a7,
        &command->a8,
        &command->a9,
    };
    *words[place] = value;
}

/* decide.c's command as the class's: its handler's number and words, field by field
 * (a struct assignment may call memset, which does not link). */
static void build_command(const wc_command* decision, km_command_t* command) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    for (uint32_t place = 0; place < 10; place++) {
        put(command, place, 0);
    }
    uint32_t handler = WORKCLUSTER_HANDLER_STEP;
    if (decision->handler == WC_HANDLER_STEP) {
        put(command, WORKCLUSTER_STEP_STATE, decision->a0);
        put(command, WORKCLUSTER_STEP_ROUND, decision->a1);
        put(command, WORKCLUSTER_STEP_FIRST, decision->first);
    } else {
        handler = WORKCLUSTER_HANDLER_SUM;
        put(command, WORKCLUSTER_SUM_STATE, decision->a0);
        put(command, WORKCLUSTER_SUM_CORES, decision->a1);
        put(command, WORKCLUSTER_SUM_FIRST, decision->first);
    }
    command->hdr = handler | decision->flags << 8;
    command->group = decision->first | decision->count << 16;
}

/* Submits `command`, waiting while the ring is full; KMX_OK and its handle, or the
 * error. */
static int submit(const km_command_t* command, uint32_t* handle) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    int error = kmx_submit(command, handle);
    while (error == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        error = kmx_submit(command, handle);
    }
    return error;
}

static void finish(uint32_t code, const wc_state* state) {
    kmx_flush(0);
    const uint32_t words[3] = {state->round, state->sum, state->trail};
    kmx_result(code, words, 3);
}

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    wc_state state;
    uint32_t words[WC_WORDS];
    for (int i = 0; i < WC_WORDS; i++) {
        words[i] = i < (int)arguments->nwords ? arguments->words[i] : 0;
    }
    uint64_t bytes;
    volatile uint32_t* groups = (volatile uint32_t*)kmx_region(0, &bytes);
    if (
        arguments->nwords < WC_WORDS || wc_start(&state, words) || groups == 0 ||
        bytes < (uint64_t)words[WC_MAX_ROUNDS] * 4
    ) {
        return 1;
    }
    for (;;) {
        wc_command decisions[2];
        km_command_t command;
        uint32_t handles[2];
        kmx_event events[2];
        wc_round(&state, decisions);
        groups[state.round] = decisions[0].first;
        kmx_progress(state.round * 4 + 1);
        /* One command at a time: SUM is exclusive, so it goes once STEP's event is
         * in. */
        for (int i = 0; i < 2; i++) {
            build_command(&decisions[i], &command);
            if (submit(&command, &handles[i]) != KMX_OK) {
                finish(3, &state);
                return 0;
            }
            kmx_progress(state.round * 4 + 2);
            const int error = kmx_wait(handles[i], kmx_time() + WAIT_TICKS, &events[i]);
            if (error != KMX_OK) {
                finish(3, &state);
                return 0;
            }
            if (events[i].kind != KM_EVENT_DONE) {
                finish(4, &state);
                return 0;
            }
        }
        kmx_progress(state.round * 4 + 3);
        const uint32_t next = wc_result(&state, events[1].ticks);
        if (next != WC_AGAIN) {
            finish(next == WC_END ? 0u : 2u, &state);
            return 0;
        }
    }
}
