/* An X280 thread that adds up the squares of 0 to n - 1, first on core 0, then on all four. */
#include "kmx.h"

#define LINE_WORDS 16 /* a 64-byte cache line holds 16 words */

// Each core stores its piece of the sum in a line of its own.
// --8<-- [start:partials]
__attribute__((section(".data"))) static uint32_t partial[4 * LINE_WORDS] = {1};
// --8<-- [end:partials]

static uint32_t squares(uint32_t first, uint32_t last) {
    uint32_t sum = 0;
    for (uint32_t i = first; i < last; i++) {
        sum += i * i;
    }
    return sum;
}

// --8<-- [start:piece]
static void piece(int core, void* argument) {
    const uint32_t n = *(const uint32_t*)argument;
    const uint32_t share = n / 4;
    partial[core * LINE_WORDS] = squares(core * share, (core + 1) * share);
}
// --8<-- [end:piece]

int kmx_main(int core) {
    (void)core;
    uint32_t n = kmx_args_get()->words[0];

    const uint64_t one_start = kmx_time();
    const uint32_t alone = squares(0, n);
    const uint64_t one_ticks = kmx_time() - one_start;

    // --8<-- [start:par]
    const uint64_t four_start = kmx_time();
    if (kmx_par(piece, &n) != KMX_OK) {
        return 1;
    }
    const uint64_t four_ticks = kmx_time() - four_start;
    // --8<-- [end:par]

    uint32_t total = 0;
    for (int h = 0; h < 4; h++) {
        total += partial[h * LINE_WORDS];
    }
    const uint32_t words[4] = {
        alone, total, (uint32_t)(one_ticks / 50), (uint32_t)(four_ticks / 50)};
    kmx_result(0, words, 4);
    return 0;
}
