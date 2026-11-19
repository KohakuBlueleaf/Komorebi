/* A copy between DRAM pages (KM_HANDLER_COPY): its argument words, where each page
 * lies and every bound, in plain C for the kernels, the thread library and tests. */
#ifndef KOMOREBI_ABI_KM_COPY_H
#define KOMOREBI_ABI_KM_COPY_H

#include <stdint.h>

#include "km_abi.h"

/* The words a0 to a9: each side's address, bank and first page, then page bytes, the
 * pages of all ranks, the cores' L1 buffer (two chunks) and a chunk's bytes. */
#define KM_COPY_FROM 0u
#define KM_COPY_FROM_BANK 1u
#define KM_COPY_FROM_FIRST 2u
#define KM_COPY_TO 3u
#define KM_COPY_TO_BANK 4u
#define KM_COPY_TO_FIRST 5u
#define KM_COPY_PAGE 6u
#define KM_COPY_PAGES 7u
#define KM_COPY_BUFFER 8u
#define KM_COPY_CHUNK 9u
/* A side's bank word: a tensor interleaved over the banks (page i in bank i % banks at
 * address + i / banks * page); else one bank's range (page i at address + i * page). */
#define KM_COPY_INTERLEAVED 0xFFu
#define KM_COPY_ALIGN 64u               /* DRAM: addresses and pages in whole lines */
#define KM_COPY_BANK_END 0x100000000ull /* a page's end in its bank: inside 4 GiB */

typedef struct {
    uint32_t address, bank, first;
} km_copy_side_t;

typedef struct {
    km_copy_side_t from, to;
    uint32_t page, pages, buffer, chunk;
} km_copy_t;

static inline km_copy_t km_copy_of(const uint32_t* a) {
    km_copy_t copy;
    copy.from.address = a[KM_COPY_FROM];
    copy.from.bank = a[KM_COPY_FROM_BANK];
    copy.from.first = a[KM_COPY_FROM_FIRST];
    copy.to.address = a[KM_COPY_TO];
    copy.to.bank = a[KM_COPY_TO_BANK];
    copy.to.first = a[KM_COPY_TO_FIRST];
    copy.page = a[KM_COPY_PAGE];
    copy.pages = a[KM_COPY_PAGES];
    copy.buffer = a[KM_COPY_BUFFER];
    copy.chunk = a[KM_COPY_CHUNK];
    return copy;
}

/* The bank and bank address of page `index` of a side, over `banks` banks. */
static inline uint64_t km_copy_at(
    const km_copy_side_t* side,
    uint32_t page,
    uint32_t index,
    uint32_t banks,
    uint32_t* bank
) {
    if (side->bank == KM_COPY_INTERLEAVED) {
        *bank = index % banks;
        return side->address + (uint64_t)(index / banks) * page;
    }
    *bank = side->bank;
    return side->address + (uint64_t)index * page;
}

/* [start, end) of a side's bytes in one bank: a range's, or the widest of a tensor's.
 */
static inline uint64_t km_copy_end(
    const km_copy_side_t* side, uint32_t page, uint32_t pages, uint32_t banks
) {
    if (side->bank == KM_COPY_INTERLEAVED) {
        uint32_t rest = 0u;
        const uint64_t rows = km_sum_divided(side->first, pages, banks, &rest);
        return side->address + (rows + (rest != 0u)) * page;
    }
    return side->address + ((uint64_t)side->first + pages) * page;
}

/* Whether a side names banks the card has, whole lines (or a tensor's reference word,
 * KM_TENSORS_MAX: an interleaved side), and pages inside its bank. */
static inline int km_copy_side_fits(
    const km_copy_side_t* side, uint32_t page, uint32_t pages, uint32_t banks
) {
    const int named = (
        side->bank == KM_COPY_INTERLEAVED && side->address < KM_TENSORS_MAX
    );
    return (
        (side->bank < banks || side->bank == KM_COPY_INTERLEAVED) &&
        (side->address % KM_COPY_ALIGN == 0u || named) &&
        km_copy_end(side, page, pages, banks) <= KM_COPY_BANK_END
    );
}

/* A side's start in one bank (km_copy_end's pair): an interleaved one's lowest. */
static inline uint64_t km_copy_start(
    const km_copy_side_t* side, uint32_t page, uint32_t banks
) {
    const uint32_t row = (
        side->bank == KM_COPY_INTERLEAVED ? side->first / banks : side->first
    );
    return side->address + (uint64_t)row * page;
}

/* Every bound of a copy by `count` ranks before anything is issued: whole lines (the
 * buffer's too), both chunks in L1, the sides in their banks and apart where shared. */
static inline int km_copy_fits(
    const km_copy_t* copy, uint32_t count, uint32_t l1_bytes, uint32_t banks
) {
    const km_copy_side_t* from = &copy->from;
    const km_copy_side_t* to = &copy->to;
    int ok = count >= 1u && count <= KM_CORES_MAX && copy->pages >= 1u;
    ok = ok && banks >= 1u && banks < KM_COPY_INTERLEAVED;
    ok = ok && copy->page >= KM_COPY_ALIGN && copy->page % KM_COPY_ALIGN == 0u;
    ok = ok && copy->chunk >= copy->page && copy->chunk % copy->page == 0u;
    ok = (
        ok && copy->buffer % KM_COPY_ALIGN == 0u &&
        (uint64_t)copy->buffer + 2ull * copy->chunk <= l1_bytes
    );
    ok = (
        ok && km_copy_side_fits(from, copy->page, copy->pages, banks) &&
        km_copy_side_fits(to, copy->page, copy->pages, banks)
    );
    const int named = (
        (from->bank == KM_COPY_INTERLEAVED && from->address < KM_TENSORS_MAX) ||
        (to->bank == KM_COPY_INTERLEAVED && to->address < KM_TENSORS_MAX)
    );
    const int shared = (
        from->bank == to->bank || from->bank == KM_COPY_INTERLEAVED ||
        to->bank == KM_COPY_INTERLEAVED
    );
    if (ok && shared && !named) {
        ok = (
            km_copy_end(from, copy->page, copy->pages, banks) <=
                km_copy_start(to, copy->page, banks) ||
            km_copy_end(to, copy->page, copy->pages, banks) <=
                km_copy_start(from, copy->page, banks)
        );
    }
    return ok;
}

#endif
