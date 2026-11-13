/* A thread that fills its ring and then stops taking events: it submits one-core
 * commands until the ring stays full for 50 ms, returns the count, and spins. */
#include <stdint.h>

#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    km_command_t r;
    r.sequence = r.land1 = r.land2 = r.land3 = 0;
    r.hdr = 1u;
    r.group = 1u << 16;
    r.a0 = r.a1 = r.a2 = r.a3 = r.a4 = r.a5 = r.a6 = r.a7 = r.a8 = r.a9 = 0;
    uint32_t submitted = 0, handle;
    uint64_t full_since = kmx_time();
    kmx_progress(1);
    while (kmx_time() - full_since < KMX_TICKS_PER_SECOND / 20 && submitted < 100000u) {
        if (kmx_submit(&r, &handle) == KMX_OK) {
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
