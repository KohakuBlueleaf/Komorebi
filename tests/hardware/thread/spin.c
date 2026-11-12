/* A thread that does nothing until it is asked to stop or its deadline passes, then
 * returns code 7 with its progress stamp at 2. */
#include <stdint.h>

#include "kmx.h"

int kmx_main(int hart) {
    (void)hart;
    kmx_progress(1);
    while (!kmx_stop_requested()) {
        ;
    }
    kmx_progress(2);
    return 7;
}
