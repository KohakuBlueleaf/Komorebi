/* Returns at once with where it ran: its island, its cluster and its flat thread
 * number (KMX_THREAD), which the host's run.threads keys it by. See ../README.md. */
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    const uint32_t words[3] = {
        (uint32_t)kmx_island(),
        (uint32_t)kmx_cluster(),
        KMX_THREAD(kmx_island(), kmx_cluster()),
    };
    kmx_result(7u, words, 3u);
    return 0;
}
