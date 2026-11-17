/* The frame driven by the class's X280 thread: vertex, bin, tally (the bins in use,
 * a result word), raster on that many cores, shade, resolve. */
#include <stdint.h>

#include "framethread_thread.h"
#include "kmx.h"

static int step(const km_command_t* command, uint32_t* word) {
    uint32_t handle;
    kmx_event event = {0u, 0u, 0u, 0u};
    if (kmx_submit(command, &handle) != KMX_OK) {
        return 1;
    }
    if (kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) != KMX_OK) {
        return 2;
    }
    *word = event.ticks;
    return event.kind == KM_EVENT_DONE ? 0 : 3;
}

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    uint32_t word = 0;
    const uint32_t cores = framethread_word(0);
    const uint32_t n = framethread_word(1);
    const uint32_t count = framethread_word(2);
    const uint32_t groups = framethread_word(3);
    framethread_vertex(&command, 0u, 0u, cores, n);
    if (step(&command, &word)) {
        return 1;
    }
    framethread_bin(&command, 0u, 0u, 1u, count, groups);
    if (step(&command, &word)) {
        return 2;
    }
    framethread_tally(&command, 0u, 0u, 1u, groups);
    if (step(&command, &word)) {
        return 3;
    }
    uint32_t active = word;
    if (active > groups) {
        return 4;
    }
    framethread_raster(&command, 0u, 0u, active);
    if (step(&command, &word)) {
        return 5;
    }
    framethread_shade(&command, 0u, 0u, cores, n);
    if (step(&command, &word)) {
        return 6;
    }
    framethread_resolve(&command, 0u, 0u, cores, n);
    if (step(&command, &word)) {
        return 7;
    }
    kmx_result(0, &active, 1);
    return 0;
}
