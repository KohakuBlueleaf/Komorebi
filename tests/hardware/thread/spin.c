/* A thread that fills region 1 (if any) with words (i + 1) * 2654435761, then waits
 * until it is asked to stop or its deadline passes; returns code 7 at stamp 2. */
#include <stdint.h>

#include "kmx.h"

int kmx_main(int hart) {
    (void)hart;
    uint64_t bytes = 0;
    uint32_t* out = kmx_region(1, &bytes);
    if (out != 0) {
        for (uint64_t i = 0; i < bytes / 4; i++) {
            out[i] = (uint32_t)(i + 1) * 2654435761u;
        }
        kmx_give(1);
    }
    kmx_progress(1);
    while (!kmx_stop_requested()) {
        ;
    }
    kmx_progress(2);
    return 7;
}
