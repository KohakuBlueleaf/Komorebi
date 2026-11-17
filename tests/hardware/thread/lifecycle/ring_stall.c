/* A thread that fills its ring and then stops taking events: it submits one-core
 * commands until the ring stays full for 50 ms, returns the count, and spins. */
#include <stdint.h>

#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = 1u;
    command.group = 1u << 16;
    command.a0 = command.a1 = command.a2 = command.a3 = command.a4 = command.a5 = 0;
    command.a6 = command.a7 = command.a8 = command.a9 = 0;
    uint32_t submitted = 0, handle;
    uint64_t full_since = kmx_time();
    kmx_progress(1);
    while (kmx_time() - full_since < KMX_TICKS_PER_SECOND / 20 && submitted < 100000u) {
        if (kmx_submit(&command, &handle) == KMX_OK) {
            submitted++;
            full_since = kmx_time();
        }
    }
    kmx_result(7, &submitted, 1);
    kmx_progress(2);
    while (!kmx_stop_requested()) {
        ;
    }
    return 7;
}
