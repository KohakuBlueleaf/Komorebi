/* The commanding thread: KM_TC_ECHO to another cluster's thread and NOP to one grid
 * core, one at a time (timed into region 0) and eight in flight; every word checked. */
#include <stdint.h>

#include "kmx.h"
#include "tc.h"

#define WAIT_TICKS (KMX_TICKS_PER_SECOND / 10)
#define WINDOW 8u /* commands in flight: the thread's command lines */

static uint32_t g_wrong __attribute__((section(".data"))) = 0;
static uint32_t g_not_done __attribute__((section(".data"))) = 0;
static uint32_t g_first_wrong __attribute__((section(".data"))) = 0xFFFFFFFFu;

/* Command `index`: KM_TC_ECHO to `cluster` with KM_FLAG_RESULT, or NOP on grid `core`.
 */
static void build(
    km_command_t* command, int echo, uint32_t index, uint32_t cluster, uint32_t core
) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = echo ? KM_TC_ECHO | KM_FLAG_RESULT << 8 : KM_HANDLER_NOP;
    command->group = echo ? KM_GROUP_CLUSTER(cluster) : core | 1u << 16;
    command->a0 = index;
    command->a1 = index ^ 0x3C3C3C3Cu;
    command->a2 = command->a3 = command->a4 = command->a5 = 0;
    command->a6 = command->a7 = command->a8 = command->a9 = 0;
}

static int submit(const km_command_t* command, uint32_t* handle) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    int status = kmx_submit(command, handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(command, handle);
    }
    return status;
}

/* Waits for `handle` and checks the event of command `index`: 0, or 1 to give up. */
static int check(uint32_t handle, int echo, uint32_t index) {
    kmx_event event;
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK) {
        return 1;
    }
    if (event.kind != KM_EVENT_DONE) {
        g_not_done++;
    } else if (echo && event.ticks != tc_echo(index, index ^ 0x3C3C3C3Cu)) {
        if (g_wrong++ == 0) {
            g_first_wrong = index;
        }
    }
    return 0;
}

/* `count` commands one at a time, each timed into `ticks`: 0, or 1 to give up. */
static int one_at_a_time(
    int echo, uint32_t count, uint32_t cluster, uint32_t core, volatile uint32_t* ticks
) {
    for (uint32_t index = 0; index < count; index++) {
        km_command_t command;
        uint32_t handle;
        build(&command, echo, index, cluster, core);
        const uint64_t started = kmx_time();
        if (submit(&command, &handle) != KMX_OK || check(handle, echo, index)) {
            return 1;
        }
        ticks[index] = (uint32_t)(kmx_time() - started);
    }
    return 0;
}

/* `count` commands with WINDOW in flight; their time in *ticks: 0, or 1 to give up. */
static int windowed(
    int echo, uint32_t count, uint32_t cluster, uint32_t core, uint32_t* ticks
) {
    uint32_t handles[WINDOW];
    const uint64_t started = kmx_time();
    for (uint32_t index = 0; index < count + WINDOW; index++) {
        if (index >= WINDOW && check(handles[index % WINDOW], echo, index - WINDOW)) {
            return 1;
        }
        if (index < count) {
            km_command_t command;
            build(&command, echo, index, cluster, core);
            if (submit(&command, &handles[index % WINDOW]) != KMX_OK) {
                return 1;
            }
        }
    }
    *ticks = (uint32_t)(kmx_time() - started);
    return 0;
}

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    uint64_t bytes;
    volatile uint32_t* ticks = (volatile uint32_t*)kmx_region(0, &bytes);
    if (arguments->nwords < KM_TC_WORDS || ticks == 0) {
        return 1;
    }
    const uint32_t count = arguments->words[KM_TC_WORD_COUNT];
    const uint32_t cluster = arguments->words[KM_TC_WORD_CLUSTER];
    const uint32_t grid_core = arguments->words[KM_TC_WORD_CORE];
    if (bytes < 8ull * count) {
        return 1;
    }
    uint32_t window_ticks[2] = {0, 0};
    int failed = 0;
    for (int echo = 1; echo >= 0 && !failed; echo--) {
        kmx_progress(echo ? 1u : 3u);
        failed = (
            one_at_a_time(echo, count, cluster, grid_core, ticks + (echo ? 0 : count))
        );
        kmx_progress(echo ? 2u : 4u);
        failed = (
            failed || windowed(echo, count, cluster, grid_core, &window_ticks[!echo])
        );
    }
    kmx_flush(0);
    const uint32_t words[5] = {
        g_wrong, g_not_done, window_ticks[0], window_ticks[1], g_first_wrong
    };
    kmx_result(failed ? 2u : 0u, words, 5);
    return 0;
}
