/* The thread of class Totals (../test_class_thread_programs.py) in a load with Plain:
 * Plain loaded, its SCALE, then TOTAL, each from totals_thread.h. */
#include "kmx.h"
#include "totals_thread.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t handle;
    kmx_event event;
    if (plain_load(0u, 4u, kmx_time() + KMX_TICKS_PER_SECOND, &event) != KMX_OK) {
        return 1;
    }
    plain_scale(&command, 0u, 0u, 4u, PLAIN_A_PAGES);
    if (kmx_submit(&command, &handle) != KMX_OK) {
        return 1;
    }
    totals_total(&command, 0u, 4u, 4u, 1u);
    return kmx_submit(&command, &handle) == KMX_OK ? 0 : 1;
}
