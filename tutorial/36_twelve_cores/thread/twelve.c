/* An X280 thread on three clusters: twelve cores add up scrambled numbers 0 to n - 1,
 * each a twelfth of them, and store their piece in a line of their cluster's region. */
#include "kmx.h"

#define LINE_WORDS 16 /* a 64-byte cache line holds 16 words */
#define CORES_PER_CLUSTER 4

// A number made from i, with no formula for the sum of them.
static uint32_t scramble(uint32_t i) {
    i ^= i >> 15;
    i *= 2246822519u;
    i ^= i >> 13;
    return i;
}

static uint32_t scrambled_sum(uint32_t first, uint32_t last) {
    uint32_t sum = 0;
    for (uint32_t i = first; i < last; i++) {
        sum += scramble(i);
    }
    return sum;
}

// --8<-- [start:piece]
static void piece(int cluster, int core, uint64_t argument) {
    const uint32_t share = (uint32_t)argument / 12;
    const uint32_t rank = (cluster - 1) * CORES_PER_CLUSTER + core;
    uint64_t bytes;
    uint32_t* pieces = (uint32_t*)kmx_region(0, &bytes);
    pieces[core * LINE_WORDS] = scrambled_sum(rank * share, (rank + 1) * share);
    kmx_flush_at(0, core * 64, 64);
}

// --8<-- [end:piece]

int kmx_main(int core) {
    (void)core;
    // --8<-- [start:serve]
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : 2;
    }
    // --8<-- [end:serve]
    const uint32_t n = kmx_args_get()->words[0];

    // --8<-- [start:all]
    uint32_t microseconds[2];
    for (int call = 0; call < 2; call++) {
        const uint64_t start = kmx_time();
        if (kmx_parallel_all(piece, n) != KMX_OK) {
            return 1;
        }
        microseconds[call] = (uint32_t)((kmx_time() - start) / 50);
    }
    // --8<-- [end:all]
    kmx_result(0, microseconds, 2);
    return 0;
}
