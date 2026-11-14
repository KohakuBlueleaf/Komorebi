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

    /* Two ranges of one bank: apart, or refused; a tensor is never compared. */
    copy = a_copy(), copy.to.bank = 0, copy.to.address = 0x10000000u + 64 * 16384;
    CHECK(km_copy_fits(&copy, 1, L1, BANKS));
    copy.to.address -= 16384;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS));
    copy.from.first = 1, copy.to.address = 0x10000000u;
    CHECK(!km_copy_fits(&copy, 1, L1, BANKS)); /* the first page moves the range */
    copy.from.bank = KM_COPY_INTERLEAVED;
    CHECK(km_copy_fits(&copy, 1, L1, BANKS));

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
    printf("ok\n");
    return 0;
}
