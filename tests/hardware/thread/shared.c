/* Clusters 2 and 3 share bank 7: 2 fills region 1 with (i + 1) * 2654435761, gives it,
 * sets region 2's flag; 3 waits for the flag (bounded), takes region 1, sums it. */
#include <stdint.h>

#include "kmx.h"

#define FLAG 0x464C4147u /* "FLAG" */

int kmx_main(int core) {
    (void)core;
    uint64_t bytes = 0, flag_bytes = 0;
    volatile uint32_t* data = kmx_region(1, &bytes);
    volatile uint32_t* flag = kmx_region(2, &flag_bytes);
    if (data == 0 || flag == 0) {
        return 1;
    }
    if (kmx_cluster() == 2) {
        for (uint64_t i = 0; i < bytes / 4; i++) {
            data[i] = (uint32_t)(i + 1) * 2654435761u;
        }
        kmx_flush(1);
        flag[0] = FLAG;
        kmx_flush(2);
        kmx_progress(2);
        return 0;
    }
    for (;;) {
        kmx_invalidate(2);
        if (flag[0] == FLAG) {
            break;
        }
        if (kmx_stop_requested()) {
            return 2;
        }
    }
    kmx_invalidate(1);
    uint32_t sum = 0;
    for (uint64_t i = 0; i < bytes / 4; i++) {
        sum += data[i];
    }
    kmx_result(0, &sum, 1);
    kmx_progress(3);
    return 0;
}
