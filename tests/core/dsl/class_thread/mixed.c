/* The thread of the merged class Mixed (../test_class_thread_chain.py): the chain
 * STEPS's records from mixed_thread.h, then its command, the word its word 0. */
#include "kmx.h"
#include "mixed_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t chain;
    uint32_t handle;
    kmx_event event;
    uint64_t bytes;
    if (
        kmx_region(MIXED_REGION_RECORDS, &bytes) == 0 ||
        bytes < MIXED_CHAIN_STEPS_BYTES ||
        mixed_chain_steps_write(MIXED_REGION_RECORDS) != KMX_OK ||
        mixed_chain_steps_command(
            &chain, MIXED_REGION_RECORDS, kmx_args_get()->words[0]
        ) != KMX_OK ||
        kmx_submit(&chain, &handle) != KMX_OK
    ) {
        return 1;
    }
    return (
        kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) == KMX_OK ? 0 : 1
    );
}
