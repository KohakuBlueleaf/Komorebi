/* An X280 thread that adds up scrambled numbers for 0 to n - 1: on core 0 alone, then
 * on all four cores. */
#include "kmx.h"

#define LINE_WORDS 16 /* a 64-byte cache line holds 16 words */

// Each core stores its piece of the sum in a line of its own.
// --8<-- [start:partials]
__attribute__((section(".data"))) static uint32_t partial[4 * LINE_WORDS] = {1};
// --8<-- [end:partials]

// --8<-- [start:sum]
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

// --8<-- [end:sum]

// --8<-- [start:piece]
static void piece(int core, void* argument) {
    const uint32_t n = *(const uint32_t*)argument;
    const uint32_t share = n / 4;
    partial[core * LINE_WORDS] = scrambled_sum(core * share, (core + 1) * share);
}

// --8<-- [end:piece]

int kmx_main(int core) {
    (void)core;
    uint32_t n = kmx_args_get()->words[0];

    const uint64_t one_start = kmx_time();
    const uint32_t alone = scrambled_sum(0, n);
    const uint64_t one_ticks = kmx_time() - one_start;

    // --8<-- [start:parallel]
    const uint64_t four_start = kmx_time();
    if (kmx_parallel(piece, &n) != KMX_OK) {
        return 1;
    }
    const uint64_t four_ticks = kmx_time() - four_start;
    // --8<-- [end:parallel]

    uint32_t total = 0;
    for (int h = 0; h < 4; h++) {
        total += partial[h * LINE_WORDS];
    }
    const uint32_t words[4] = {
        alone, total, (uint32_t)(one_ticks / 50), (uint32_t)(four_ticks / 50)
    };
    kmx_result(0, words, 4);
    return 0;
}
