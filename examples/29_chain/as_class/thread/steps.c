/* The chain STEPS run by the class's X280 thread: its records written into the region
 * `records` by threadsteps_thread.h, then its command. See ../../README.md. */
#include <stdint.h>

#include "kmx.h"
#include "threadsteps_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t chain;
    uint32_t handle;
    kmx_event event = {0u, 0u, 0u, 0u};
    uint64_t bytes;
    uint32_t stage = 1u; /* the step that failed: 1 region, 2 records, 3 command ... */
    if (
        kmx_region(THREADSTEPS_REGION_RECORDS, &bytes) != 0 &&
        bytes >= THREADSTEPS_CHAIN_STEPS_BYTES && (stage = 2u) &&
        threadsteps_chain_steps_write(THREADSTEPS_REGION_RECORDS) == KMX_OK &&
        (stage = 3u) &&
        threadsteps_chain_steps_command(&chain, THREADSTEPS_REGION_RECORDS, 0u, 0u) ==
            KMX_OK &&
        (stage = 4u) && kmx_submit(&chain, &handle) == KMX_OK && (stage = 5u) &&
        kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) == KMX_OK
    ) {
        stage = 0u;
    }
    const uint32_t words[3] = {stage, event.kind, event.ticks};
    kmx_result(stage, words, 3);
    return (int)stage;
}
