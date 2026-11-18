/* km_copy_fits on the host: overlapping sides refused (one bank, two interleaved, a
 * range inside an interleaved tensor), apart ones taken, the buffer in whole 64 B. */
#include <stdio.h>

#include "komorebi/abi/km_copy.h"

static int fits(
    uint32_t fa,
    uint32_t fb,
    uint32_t ff,
    uint32_t ta,
    uint32_t tb,
    uint32_t tf,
    uint32_t pages,
    uint32_t buffer
) {
    km_copy_t c = {{fa, fb, ff}, {ta, tb, tf}, 2048u, pages, buffer, 8192u};
    return km_copy_fits(&c, 4u, 1u << 20, 8u);
}

int main(void) {
    const uint32_t I = KM_COPY_INTERLEAVED, A = 1u << 24, B = 2u << 24;

    struct {
        const char* name;
        int got, want;
    } rows[] = {
        {"two tensors apart", fits(A, I, 0, B, I, 0, 64, 0x20000), 1},
        {"one tensor shifted a page", fits(A, I, 0, A, I, 1, 64, 0x20000), 0},
        {"one tensor onto itself", fits(A, I, 0, A, I, 0, 64, 0x20000), 0},
        {"one tensor, pages apart", fits(A, I, 0, A, I, 64, 64, 0x20000), 1},
        {"a bank range inside a tensor", fits(A, I, 0, A + 1024, 3, 0, 4, 0x20000), 0},
        {"a bank range after its rows", fits(A, I, 0, A + 2048, 3, 0, 4, 0x20000), 1},
        {"a bank range past a tensor", fits(A, I, 0, B, 3, 0, 4, 0x20000), 1},
        {"one bank, overlapping", fits(A, 2, 0, A + 2048, 2, 0, 4, 0x20000), 0},
        {"one bank, apart", fits(A, 2, 0, A + 8192, 2, 0, 4, 0x20000), 1},
        {"two banks, same addresses", fits(A, 2, 0, A, 5, 0, 4, 0x20000), 1},
        {"a named tensor word (not compared)", fits(3, I, 0, 3, I, 1, 64, 0x20000), 1},
        {"buffer 16-aligned only", fits(A, I, 0, B, I, 0, 64, 0x20010), 0},
    };

    int bad = 0;
    for (unsigned i = 0; i < sizeof rows / sizeof rows[0]; i++) {
        const int ok = rows[i].got == rows[i].want;
        bad += !ok;
        printf("%s %s: %d\n", ok ? "ok  " : "FAIL", rows[i].name, rows[i].got);
    }
    printf(bad ? "FAILED %d\n" : "all ok\n", bad);
    return bad != 0;
}
