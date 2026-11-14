/* Thread load example, the thread: ROUNDS moves the boundary between sum's cores and
 * widen's each round with a load, then commands both; CHECK shows the refusals. */
#include <stdint.h>

#include "thread_load.h"
#include "kmx.h"

/* a command's wait, and a full ring's retry */
#define WAIT_TICKS KMX_TICKS_PER_SECOND
#define LONG_TILES 128u /* CHECK's long SUM on one core */

/* Submits `command`, retrying a full ring: KMX_OK and its handle, or why it failed. */
static int submit(const km_command_t* command, uint32_t* handle) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status;
    do {
        status = kmx_submit(command, handle);
    } while (status == KMX_FULL && kmx_time() < deadline);
    return status;
}

/* Submits `command` and waits for its event. */
static int run(const km_command_t* command, kmx_event* event) {
    uint32_t handle = 0;
    const int status = submit(command, &handle);
    return status != KMX_OK ? status : kmx_wait(handle, kmx_time() + WAIT_TICKS, event);
}

/* A command of `program` (SUM or WIDEN, handler 1 of each) on cores [first, first +
 * count): `tiles` from input tile `from`, its lines from line `line` of the outs. */
static void work(
    km_command_t* command,
    uint32_t program,
    uint32_t first,
    uint32_t count,
    uint32_t tiles,
    uint32_t from,
    uint32_t line,
    uint32_t tag
) {
    const uint32_t* words = kmx_args_get()->words;
    kmx_command(command, program, TL_HANDLER_SUM, first, count);
    command->a0 = words[TL_WORD_INPUT];
    command->a1 = tiles;
    command->a2 = words[TL_WORD_OUTS];
    command->a3 = line;
    command->a4 = tag;
    command->a6 = from;
}

/* One job: each round its boundary, a load of the cores that change program, both
 * commands, both waits; each round's log line in region 1. */
static uint32_t rounds(const km_command_t* job) {
    const uint32_t cores = kmx_args_get()->words[TL_WORD_CORES];
    uint64_t round_bytes = 0, log_bytes = 0;
    const uint32_t* plan = (const uint32_t*)kmx_region(0, &round_bytes);
    uint32_t* log = (uint32_t*)kmx_region(1, &log_bytes);
    const uint32_t first = job->a0, count = job->a1;
    uint32_t k = job->a3;
    if (
        cores < 2 || k < 1 || k >= cores ||
        4ull * TL_ROUND * (first + count) > round_bytes ||
        4ull * TL_LOG * count > log_bytes
    ) {
        return KMX_BAD;
    }
    kmx_invalidate(0); /* the host's amounts, not lines a cache kept from before */
    int status = KMX_OK;
    for (uint32_t round = first; round < first + count && status == KMX_OK; round++) {
        const uint64_t started = kmx_time();
        const uint32_t* amounts = plan + TL_ROUND * round;
        const uint32_t sum_tiles = amounts[TL_ROUND_SUM];
        const uint32_t widen_tiles = amounts[TL_ROUND_WIDEN];
        uint32_t split = TL_SPLIT(cores, sum_tiles, widen_tiles);
        split = split < 1 ? 1 : split > cores - 1 ? cores - 1 : split;
        uint32_t* line = log + TL_LOG * (round - first);
        line[TL_LOG_LOADS] = line[TL_LOG_LOAD_TICKS] = line[TL_LOG_LOAD_EVENT] = 0;
        if (split != k) { /* the cores between the two boundaries change program */
            kmx_event loaded;
            loaded.ticks = 0;
            const uint64_t asked = kmx_time();
            status = (
                split < k
                    ? kmx_load(
                        kProgramWiden, split, k - split, asked + WAIT_TICKS, &loaded
                    )
                    : kmx_load(kProgramSum, k, split - k, asked + WAIT_TICKS, &loaded)
            );
            line[TL_LOG_LOADS] = 1;
            line[TL_LOG_LOAD_TICKS] = (uint32_t)(kmx_time() - asked);
            line[TL_LOG_LOAD_EVENT] = loaded.ticks;
            k = status == KMX_OK ? split : k;
        }
        km_command_t sum, widen;
        const uint32_t tag = job->a2 << 16 | (round & 0xFFFFu);
        work(
            &sum,
            kProgramSum,
            0,
            k,
            sum_tiles,
            amounts[TL_ROUND_SUM_FIRST],
            round * cores,
            tag
        );
        work(
            &widen,
            kProgramWiden,
            k,
            cores - k,
            widen_tiles,
            amounts[TL_ROUND_WIDEN_FIRST],
            round * cores,
            tag
        );
        uint32_t sum_handle = 0, widen_handle = 0;
        kmx_event sum_event, widen_event;
        if (status == KMX_OK) {
            status = submit(&sum, &sum_handle);
        }
        if (status == KMX_OK) {
            status = submit(&widen, &widen_handle);
        }
        if (status == KMX_OK) {
            status = kmx_wait(sum_handle, kmx_time() + WAIT_TICKS, &sum_event);
        }
        if (status == KMX_OK) {
            status = kmx_wait(widen_handle, kmx_time() + WAIT_TICKS, &widen_event);
        }
        if (
            status == KMX_OK &&
            (sum_event.kind != KM_EVENT_DONE || widen_event.kind != KM_EVENT_DONE)
        ) {
            status = KMX_LATE;
        }
        line[TL_LOG_SPLIT] = k;
        line[TL_LOG_SUM_TICKS] = status == KMX_OK ? sum_event.ticks : 0;
        line[TL_LOG_WIDEN_TICKS] = status == KMX_OK ? widen_event.ticks : 0;
        line[TL_LOG_ROUND_TICKS] = (uint32_t)(kmx_time() - started);
        line[TL_LOG_STATUS] = (uint32_t)status;
    }
    kmx_flush(1);
    return (uint32_t)status;
}

/* A SUM to a core of widen (refused), a load to a core of sum while its long SUM runs
 * (rejected busy, then kmx_load), and back; the findings in region 2. */
static uint32_t check(const km_command_t* command) {
    uint64_t bytes = 0;
    uint32_t* out = (uint32_t*)kmx_region(2, &bytes);
    const uint32_t sum_core = command->a1, widen_core = command->a2;
    if (bytes < 4u * TL_CHECK) {
        return KMX_BAD;
    }
    km_command_t sum, load;
    kmx_event event;
    event.kind = event.auxiliary = event.ticks = 0;
    work(&sum, kProgramSum, widen_core, 1, 1, 0, command->a3, 0);
    int status = run(&sum, &event);
    out[TL_CHECK_REFUSED] = status == KMX_OK ? event.kind << 8 | event.auxiliary : ~0u;
    work(&sum, kProgramSum, sum_core, 1, LONG_TILES, 0, command->a3, 0);
    uint32_t long_handle = 0, handle = 0;
    const uint64_t asked = kmx_time();
    status = submit(&sum, &long_handle);
    kmx_command(&load, kProgramWiden, KM_HANDLER_LOAD, sum_core, 1);
    load.a0 = command->a0; /* widen's record, from the host */
    if (status == KMX_OK) {
        status = submit(&load, &handle);
    }
    if (status == KMX_OK) {
        status = kmx_wait(handle, kmx_time() + WAIT_TICKS, &event);
    }
    out[TL_CHECK_BUSY] = status == KMX_OK ? event.kind << 8 | event.auxiliary : ~0u;
    out[TL_CHECK_LOAD_STATUS] = (
        (uint32_t)kmx_load(kProgramWiden, sum_core, 1, kmx_time() + WAIT_TICKS, &event)
    );
    out[TL_CHECK_LOAD_TICKS] = (uint32_t)(kmx_time() - asked);
    status = kmx_wait(long_handle, kmx_time() + WAIT_TICKS, &event);
    out[TL_CHECK_SUM_TICKS] = status == KMX_OK ? event.ticks : ~0u;
    out[TL_CHECK_BACK_STATUS] = (
        (uint32_t)kmx_load(kProgramSum, sum_core, 1, kmx_time() + WAIT_TICKS, &event)
    );
    work(&sum, kProgramSum, sum_core, 1, 1, 0, command->a3, 0);
    status = run(&sum, &event);
    out[TL_CHECK_AFTER] = status == KMX_OK ? event.kind : ~0u;
    out[TL_CHECK_LONG_TILES] = LONG_TILES;
    kmx_flush(2);
    return 0;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerRounds:
            return rounds(command);
        case kHandlerCheck:
            return check(command);
    }
    return KMX_BAD;
}
