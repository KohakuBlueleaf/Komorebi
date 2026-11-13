/* An X280 thread that reads the numbers Python put in a region and writes three times
 * each into another region. */
#include "kmx.h"

int kmx_main(int core) {
    (void)core;

    // --8<-- [start:regions]
    uint64_t numbers_bytes;
    uint64_t tripled_bytes;
    const uint32_t* numbers = (const uint32_t*)kmx_region(0, &numbers_bytes);
    uint32_t* tripled = (uint32_t*)kmx_region(1, &tripled_bytes);
    // --8<-- [end:regions]
    if (numbers == 0 || tripled == 0) {
        return 1;
    }

    // --8<-- [start:work]
    kmx_invalidate(0);
    for (uint64_t i = 0; i < numbers_bytes / 4; i++) {
        tripled[i] = 3 * numbers[i];
    }
    kmx_flush(1);
    // --8<-- [end:work]

    const uint32_t count = (uint32_t)(numbers_bytes / 4);
    kmx_result(0, &count, 1);
    return 0;
}
