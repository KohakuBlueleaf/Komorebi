/* The thread of class DirectThreaded (../test_class_thread_chain.py): the chain
 * SUMMED's records from directthreaded_thread.h, then its command. */
#include "kmx.h"
#include "directthreaded_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t chain;
    uint32_t handle;
    kmx_event event;
    uint64_t bytes;
    if (
        kmx_region(DIRECTTHREADED_REGION_RECORDS, &bytes) == 0 ||
        bytes < DIRECTTHREADED_CHAIN_SUMMED_BYTES ||
        directthreaded_chain_summed_write(DIRECTTHREADED_REGION_RECORDS) != KMX_OK ||
        directthreaded_chain_summed_command(
            &chain, DIRECTTHREADED_REGION_RECORDS, 0u
        ) != KMX_OK ||
        kmx_submit(&chain, &handle) != KMX_OK
    ) {
        return 1;
    }
    return (
        kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) == KMX_OK ? 0 : 1
    );
}
