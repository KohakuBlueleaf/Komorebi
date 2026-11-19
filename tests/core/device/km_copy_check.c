/* A copy's bounds and where its pages lie (komorebi/abi/km_copy.h) on the CPU; prints
 * "ok" or the first failed check. Built and run by test_copy.py. */
#include <stdint.h>
#include <stdio.h>

#include "komorebi/abi/km_copy.h"

#define CHECK(condition)                               \
    if (!(condition)) {                                \
        printf("line %d: %s\n", __LINE__, #condition); \
        return 1;                                      \
    }

#define L1 0x180000u
#define BANKS 8u

/* 64 pages of 16 KiB from bank 0 into bank 6, through a buffer at 0x40000. */
static km_copy_t a_copy(void) {
    const uint32_t words[10] = {
        0x10000000u, 0, 0, 0x20000000u, 6, 0, 16384, 64, 0x40000, 16384
    };
    return km_copy_of(words);
}

int main(void) {
    km_copy_t copy = a_copy();
    CHECK(km_copy_fits(&copy, 12, L1, BANKS));
    CHECK(copy.from.bank == 0 && copy.to.bank == 6 && copy.pages == 64);

    /* Each bound refuses alone. */
    copy = a_copy(), copy.page = 48, copy.chunk = 48 * 64;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* not whole lines */
    copy = a_copy(), copy.chunk = 16384 + 64;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* not whole pages */
    copy = a_copy(), copy.buffer = L1 - 16384;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* the second chunk past L1 */
    copy = a_copy(), copy.to.bank = BANKS;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* a bank the card lacks */
    copy = a_copy(), copy.from.address = 0xFFFF0000u;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* past 4 GiB of the bank */
    copy = a_copy(), copy.to.address += 32;
    CHECK(!km_copy_fits(&copy, 12, L1, BANKS)); /* not a whole line */
    copy = a_copy();
    CHECK(!km_copy_fits(&copy, 0, L1, BANKS) && !km_copy_fits(&copy, 129, L1, BANKS));
    copy.pages = 0;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS));

    /* Two sides that share a bank (one bank, or an interleaved tensor's): apart, or
     * refused. */
    copy = a_copy(), copy.to.bank = 0, copy.to.address = 0x10000000u + 64 * 16384;
    CHECK(km_copy_fits(&copy, 1, L1, BANKS));
    copy.to.address -= 16384;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS));
    copy.from.first = 1, copy.to.address = 0x10000000u;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS)); /* the first page moves the range */
    copy.from.bank = KM_COPY_INTERLEAVED;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS)); /* the tensor's pages in bank 0 too */

    /* Pages: a range's in one bank one after another, a tensor's dealt over the
     * banks, its end in a bank the widest of them. */
    km_copy_side_t range = {0x1000, 3, 2}, tensor = {0x1000, KM_COPY_INTERLEAVED, 5};
    uint32_t bank = 0;
    CHECK(km_copy_at(&range, 4096, 2 + 7, BANKS, &bank) == 0x1000 + 9 * 4096);
    CHECK(bank == 3);
    CHECK(km_copy_at(&tensor, 4096, 5 + 13, BANKS, &bank) == 0x1000 + 2 * 4096);
    CHECK(bank == 2);
    CHECK(km_copy_end(&range, 4096, 7, BANKS) == 0x1000 + 9 * 4096);
    CHECK(km_copy_end(&tensor, 4096, 12, BANKS) == 0x1000 + 3 * 4096);

    /* (a + b) / d by 32-bit divisions: as the 64-bit one, at the words' edges too
     * (Wormhole's 12 banks: no shift). */
    const uint32_t words[] = {
        0, 1, 7, 11, 12, 13, 0x7FFFFFFFu, 0xFFFFFFF4u, 0xFFFFFFFFu
    };
    const uint32_t divisors[] = {1, 2, 8, 12, 254, 0x80000001u, 0xFFFFFFFFu};
    for (unsigned i = 0; i < sizeof words / sizeof *words; i++) {
        for (unsigned j = 0; j < sizeof words / sizeof *words; j++) {
            for (unsigned k = 0; k < sizeof divisors / sizeof *divisors; k++) {
                const uint64_t sum = (uint64_t)words[i] + words[j];
                uint32_t rest = 0;
                const uint64_t q = (
                    km_sum_divided(words[i], words[j], divisors[k], &rest)
                );
                CHECK(q == sum / divisors[k] && rest == sum % divisors[k]);
            }
        }
    }
    const km_copy_side_t far = {0, KM_COPY_INTERLEAVED, 0xFFFFFFF0u};
    CHECK(km_copy_end(&far, 64, 0xFFFFFFF0u, 12) == (0x1FFFFFFE0ull + 11) / 12 * 64);
    CHECK(
        km_copy_at(&tensor, 64, 0xFFFFFFFFu, 12, &bank) == 0x1000 + 0x15555555ull * 64
    );
    CHECK(bank == 3);
    printf("ok\n");
    return 0;
}
