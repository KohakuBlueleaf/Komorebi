/* A thread that fills its ring and then stops taking events: it issues one-core
 * records until the ring stays full for 50 ms, returns the count, and spins. */
#include <stdint.h>

#include "kmx.h"

int kmx_main(int hart) {
    (void)hart;
    km_record_t r;
    r.seq = r.land1 = r.land2 = r.land3 = 0;
    r.hdr = 1u;
    r.team = 1u << 16;
    r.a0 = r.a1 = r.a2 = r.a3 = r.a4 = r.a5 = r.a6 = r.a7 = r.a8 = r.a9 = 0;
    uint32_t issued = 0, ticket;
    uint64_t full_since = kmx_time();
    kmx_progress(1);
    while (kmx_time() - full_since < KMX_TICKS_PER_SECOND / 20 && issued < 100000u) {
        if (kmx_issue(&r, &ticket) == KMX_OK) {
            issued++;
            full_since = kmx_time();
        }
    }
    kmx_result(7, &issued, 1);
    kmx_progress(2);
    while (!kmx_stop_requested()) {
        ;
    }
    return 7;
}
