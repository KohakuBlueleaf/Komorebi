/* A thread naming a handler and a region its class lacks: the make refuses it
 * (../test_class_thread_header.py). */
#include "kmx.h"
#include "missing_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    missing_nothing(&command, 0u, 0u, 4u);
    kmx_region(MISSING_REGION_ITEMS, 0);
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 1;
}
