/* Over islands, the X280 ranks' counts for the run line's handler word (plain C): the
 * message ranks of one island's part of a group and of the group over every part. */
#pragma once

#include <stdint.h>

/* `entries`: 4 words an island, word 0 its flat ranks (first | count << 16); islands
 * `low` to `high` hold `group`, each part less its `counters` counter tiles. Island
 * `here`'s part in bits 8 to 15, the group's in bits 16 to 31. */
static inline uint32_t kmd_x280_counts(
    const uint32_t* entries,
    uint32_t low,
    uint32_t high,
    uint32_t here,
    uint32_t group,
    uint32_t counters
) {
    const uint32_t first = group & 0xFFFFu, end = first + (group >> 16);
    uint32_t total = 0, own = 0;
    for (uint32_t island = low; island <= high; island++) {
        const uint32_t range = entries[4u * island];
        const uint32_t start_at = range & 0xFFFFu, stop_at = start_at + (range >> 16);
        const uint32_t start = first > start_at ? first : start_at;
        const uint32_t stop = end < stop_at ? end : stop_at;
        const uint32_t held = stop > start + counters ? stop - start - counters : 0u;
        total += held;
        own = island == here ? held : own;
    }
    return own << 8 | total << 16;
}
