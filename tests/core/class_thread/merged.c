/* The thread of the merged class Countedrain (../test_class_thread_header.py): the
 * merged numbers; MergedCounted's TOTAL and Drain's DRAIN by their classes' names. */
#include "countedrain_thread.h"
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    uint64_t bytes;
    if (
        kmx_region(COUNTEDRAIN_REGION_COUNTS, &bytes) == 0 || bytes < MERGEDCOUNTED_SIZE
    ) {
        return 1;
    }
    mergedcounted_total(&command, 0u, 0u, 4u, 1u);
    if (kmx_submit(&command, &handle) != KMX_OK) {
        return 1;
    }
    drain_drain(&command, 0u, 0u, 4u);
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 1;
}
