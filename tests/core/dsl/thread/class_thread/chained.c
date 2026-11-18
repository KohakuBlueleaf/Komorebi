/* The thread of class Threaded (../test_class_thread_chain.py): the chain STEPS's
 * records from threaded_thread.h, then its command, the word the thread's word 0. */
#include "kmx.h"
#include "threaded_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t chain;
    uint32_t handle;
    kmx_event event;
    uint64_t bytes;
    if (
        kmx_region(THREADED_REGION_RECORDS, &bytes) == 0 ||
        bytes < THREADED_CHAIN_STEPS_BYTES ||
        threaded_chain_steps_write(THREADED_REGION_RECORDS) != KMX_OK ||
        threaded_chain_steps_command(
            &chain, THREADED_REGION_RECORDS, kmx_args_get()->words[0], 0u
        ) != KMX_OK ||
        kmx_submit(&chain, &handle) != KMX_OK
    ) {
        return 1;
    }
    return (
        kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) == KMX_OK ? 0 : 1
    );
}
