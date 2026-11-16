/* The thread of class Counted (../test_class_thread_header.py): TOTAL's command filled
 * by counted_thread.h, its region the class's own. */
#include "counted_thread.h"
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    uint64_t bytes;
    if (kmx_region(COUNTED_REGION_COUNTS, &bytes) == 0 || bytes < COUNTED_SIZE) {
        return 1;
    }
    counted_total(&command, 0u, 0u, 4u, 1u);
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 1;
}
