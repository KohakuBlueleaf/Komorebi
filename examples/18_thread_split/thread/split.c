/* Thread split example, the thread: seven passes over the grid's ports (split.h); in
 * each the cores pull ranges and every answer's word names the overlay to run on it. */
#include <stdint.h>

#include "split.h"
#include "kmx.h"

#define CODE_BAD 1u  /* words out of range */
#define CODE_LATE 2u /* a pass was late or stopped, or the ring stayed full */
#define FULL_TICKS KMX_TICKS_PER_SECOND

/* The thread's choice for each core in the thread's pass, from the core's last tell. */
static uint32_t g_choice[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};

/* Core's index-th range: the cores split the ranges into equal stripes in core order.
 */
static uint32_t stripe(int core, uint32_t index, uint32_t ranges) {
    const uint64_t cores = (uint64_t)kmx_port_cores();
    const uint32_t first = (uint32_t)((uint64_t)core * ranges / cores);
    const uint32_t end = (uint32_t)((uint64_t)(core + 1) * ranges / cores);
    return first + index < end ? first + index : KMX_PULL_NONE;
}

/* The overlay of the kind the core's newest tell counted most units of. */
static uint32_t thread_choice(int core, uint32_t index) {
    kmx_message told;
    const int fresh = kmx_port_told(core, &told) == KMX_OK;
    if (index == 0) { /* a tell left from the pass before: not this pass's */
        g_choice[core] = TS_OVERLAY_PLAIN;
    } else if (fresh) {
        uint32_t best = 0;
        for (uint32_t kind = 1; kind < TS_OVERLAYS; kind++) {
            if ((told.w[0] >> 8 * kind & 0xFFu) > (told.w[0] >> 8 * best & 0xFFu)) {
                best = kind;
            }
        }
        g_choice[core] = TS_OVERLAY_PLAIN + best;
    }
    return g_choice[core];
}

/* --8<-- [start:pick] */
/* arg: the ranges, the pass << 32. The range and the overlay for it. */
static uint64_t pick_word(int core, uint32_t index, uint64_t arg) {
    const uint32_t pass = (uint32_t)(arg >> 32);
    const uint32_t range = stripe(core, index, (uint32_t)arg);
    uint32_t word = TS_OVERLAY_PLAIN + pass;
    if (pass == TS_PASS_HOST) {
        const uint32_t* words = kmx_args_get()->words;
        word = words[TS_WORD_SPLIT + core / 16] >> 2 * (core % 16) & 3u;
    } else if (pass == TS_PASS_THREAD) {
        word = thread_choice(core, index);
    }
    return KMX_PULL_WORD(range, word);
}

/* --8<-- [end:pick] */

/* The pull-only passes: the same ranges without a word, and with a fixed one. */
static uint32_t pick(int core, uint32_t index, uint64_t arg) {
    return stripe(core, index, (uint32_t)arg);
}

static uint64_t pick_fixed_word(int core, uint32_t index, uint64_t arg) {
    return KMX_PULL_WORD(stripe(core, index, (uint32_t)arg), TS_PULL_WORD);
}

/* Runs one pass until its event, retrying a full ring; its ticks into *elapsed. */
static int run_pass(
    const km_command_t* command, const kmx_pull* pull, uint32_t* elapsed
) {
    const uint64_t start = kmx_time(), deadline = start + FULL_TICKS;
    kmx_event event;
    int error;
    do {
        error = kmx_pull_run(command, pull, &event);
    } while (error == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested());
    *elapsed = (uint32_t)(kmx_time() - start);
    return error == KMX_OK && event.kind != KM_EVENT_DONE ? KMX_LATE : error;
}

/* The seven passes, their ticks into ticks[]; the result code. */
static uint32_t passes(uint32_t ticks[TS_PASSES]) {
    const kmx_args* arguments = kmx_args_get();
    const uint32_t* words = arguments->words;
    if (
        arguments->nwords < TS_WORDS || kmx_port_cores() == 0 ||
        words[TS_WORD_PER] == 0 || words[TS_WORD_RUN] == 0
    ) {
        return CODE_BAD;
    }
    for (uint32_t pass = 0; pass < TS_PASSES; pass++) {
        const int work = pass < TS_WORK_PASSES;
        const uint32_t units = work ? words[TS_WORD_UNITS] : words[TS_WORD_PULL_UNITS];
        const uint32_t per = work ? words[TS_WORD_PER] : 1u;
        const uint64_t ranges = ((uint64_t)units + per - 1) / per;
        const kmx_pull pull = {
            units,
            per,
            KMX_PULL_COUNT,
            0u,
            pass == TS_PASS_PICK ? pick : 0,
            ranges | (uint64_t)pass << 32,
            pass == TS_PASS_PICK       ? 0
                : pass == TS_PASS_WORD ? pick_fixed_word
                                       : pick_word
        };
        /* field by field: a struct assignment may call memset, which does not link */
        km_command_t command;
        command.sequence = command.land1 = command.land2 = command.land3 = 0;
        command.hdr = TS_HANDLER_PASS;
        command.group = 0 | (uint32_t)kmx_port_cores() << 16;
        command.a0 = units;
        command.a1 = words[TS_WORD_TILES];
        command.a2 = work ? pass * words[TS_WORD_UNITS] : 0;
        command.a3 = words[TS_WORD_RUN];
        command.a4 = words[TS_WORD_PENALTY];
        command.a5 = words[TS_WORD_LINES];
        command.a6 = pass;
        command.a7 = !work;
        command.a8 = command.a9 = 0;
        const int error = run_pass(&command, &pull, &ticks[pass]);
        if (error != KMX_OK) {
            return error == KMX_BAD ? CODE_BAD : CODE_LATE;
        }
    }
    return 0;
}

int kmx_main(int x280_core) {
    (void)x280_core;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
#if defined(TS_STARTED_BY_HOST)
    return 0; /* the far island and the links run only once the load returns */
#else
    uint32_t ticks[TS_PASSES];
    const uint32_t code = passes(ticks);
    if (code == 0) {
        kmx_result(0, ticks, TS_PASSES);
    }
    return (int)code;
#endif
}

#if defined(TS_STARTED_BY_HOST)
uint32_t kmx_program(const km_command_t* command) {
    if (kmx_handler(command) != TS_THREAD_PASSES) {
        return CODE_BAD;
    }
    uint64_t bytes = 0;
    uint32_t* ticks = (uint32_t*)kmx_region(0, &bytes);
    if (ticks == 0 || bytes < 4 * TS_PASSES) {
        return CODE_BAD;
    }
    const uint32_t code = passes(ticks);
    kmx_flush(0); /* the host reads the ticks */
    return code;
}
#endif
