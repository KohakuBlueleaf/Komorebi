/* The thread added to class Sums after its make (../test_class_thread_header.py):
 * TOTAL's command filled by sums_thread.h. */
#include "kmx.h"
#include "sums_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    sums_total(&command, 0u, 0u, 4u, 1u);
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 1;
}
