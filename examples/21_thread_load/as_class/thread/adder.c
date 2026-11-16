/* Adder's thread in a load with Doubler: Doubler loaded on Adder's cores, DOUBLE on all
 * eight, Adder loaded back, ADD on its four; each from adder_thread.h. */
#include <stdint.h>

#include "kmx.h"
#include "adder_thread.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* a step's wait: shorter than any stall */

/* Submits `command` and waits for its event: KMX_OK when it is done. */
static int run(const km_command_t* command) {
    uint32_t handle;
    kmx_event event;
    if (kmx_submit(command, &handle) != KMX_OK) {
        return KMX_BAD;
    }
    if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK) {
        return KMX_LATE;
    }
    return event.kind == KM_EVENT_DONE ? KMX_OK : KMX_BAD;
}

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    kmx_event event;
    const uint32_t width = kmx_args_get()->words[0];
    uint32_t stage = 1u; /* the step that failed: 1 load, 2 double, 3 load back, 4 add */
    if (doubler_load(0u, 4u, kmx_time() + WAIT_TICKS, &event) == KMX_OK) {
        stage = 2u;
        doubler_double(&command, 0u, 0u, 8u, width);
        if (run(&command) == KMX_OK) {
            stage = 3u;
            if (adder_load(0u, 4u, kmx_time() + WAIT_TICKS, &event) == KMX_OK) {
                stage = 4u;
                adder_add(&command, 0u, 0u, 4u, 2u * width);
                stage = run(&command) == KMX_OK ? 0u : 4u;
            }
        }
    }
    const uint32_t words[1] = {stage};
    kmx_result(stage, words, 1);
    return (int)stage;
}
