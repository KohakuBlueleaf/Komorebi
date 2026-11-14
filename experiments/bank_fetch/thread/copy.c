/* The copy's rate: for each count of Tensix cores, region 0 cleared, then `repeats`
 * kmx_fetch calls of a range of another bank into it, timed, then its hash; a store. */
#include <stdint.h>

#include "kmx.h"

/* words */
#define W_BANK 0u    /* the range's bank */
#define W_ADDRESS 1u /* its bank address */
#define W_BYTES 2u
#define W_REPEATS 3u
#define W_COUNTS 4u /* how many counts of cores follow, at most COUNTS_MAX */
#define W_COUNT 5u
#define W_STORE 9u /* bytes stored back with kmx_store at W_STORE_ADDRESS (0: none) */
#define W_STORE_ADDRESS 10u
#define COUNTS_MAX 4u
/* result words: per count i at 8 i, then the store's at 32 */
#define R_CALL 0u  /* the best call's ticks, kmx_fetch's call to its return */
#define R_EVENT 1u /* its event's ticks, the dispatcher's submit to completion */
#define R_HASH 2u  /* two words: sum of 64-bit word j times 2 j + 1 */
#define R_STATUS 4u
#define R_WORST 5u
#define R_STORE 32u /* status, call ticks */
#define RESULT_WORDS 34u

/* The region's hash, as the host's numpy computes it. */
static uint64_t hash(const volatile uint64_t* words, uint64_t count) {
    uint64_t sum = 0;
    for (uint64_t j = 0; j < count; j++) {
        sum += words[j] * (2 * j + 1);
    }
    return sum;
}

int kmx_main(int core) {
    (void)core;
    const uint32_t* words = kmx_args_get()->words;
    uint32_t result[RESULT_WORDS] __attribute__((aligned(8)));
    for (uint32_t i = 0; i < RESULT_WORDS; i++) {
        result[i] = 0;
    }
    uint64_t region_bytes = 0;
    volatile uint64_t* region = (volatile uint64_t*)kmx_region(0, &region_bytes);
    const uint64_t bytes = words[W_BYTES];
    if (region == 0 || bytes > region_bytes || words[W_COUNTS] > COUNTS_MAX) {
        return 1;
    }
    const kmx_dram range = {words[W_ADDRESS], words[W_BANK], 0, 0};
    for (uint32_t i = 0; i < words[W_COUNTS]; i++) {
        const uint32_t count = words[W_COUNT + i];
        uint32_t* out = &result[8 * i];
        kmx_progress(1 + i);
        for (uint64_t j = 0; j < bytes / 8; j++) {
            region[j] = 0;
        }
        kmx_flush_at(0, 0, bytes);
        out[R_CALL] = 0xFFFFFFFFu;
        for (uint32_t r = 0; r < words[W_REPEATS]; r++) {
            kmx_event event = {0, 0, 0, 0};
            const uint64_t start = kmx_time();
            const int status = (
                kmx_fetch(
                    0, 0, &range, bytes, 0, count, start + KMX_TICKS_PER_SECOND, &event
                )
            );
            const uint32_t ticks = (uint32_t)(kmx_time() - start);
            if (status != KMX_OK) {
                out[R_STATUS] = (
                    (uint32_t)status | event.kind << 8 | event.auxiliary << 16
                );
                break;
            }
            if (ticks < out[R_CALL]) {
                out[R_CALL] = ticks;
                out[R_EVENT] = event.ticks;
            }
            out[R_WORST] = ticks > out[R_WORST] ? ticks : out[R_WORST];
        }
        const uint64_t sum = hash(region, bytes / 8);
        out[R_HASH] = (uint32_t)sum;
        out[R_HASH + 1] = (uint32_t)(sum >> 32);
    }
    if (words[W_STORE] != 0) {
        kmx_progress(9);
        const kmx_dram to = {words[W_STORE_ADDRESS], words[W_BANK], 0, 0};
        kmx_event event = {0, 0, 0, 0};
        const uint64_t start = kmx_time();
        result[R_STORE] = (
            (uint32_t)kmx_store(
                0, 0, &to, words[W_STORE], 0, 4, start + KMX_TICKS_PER_SECOND, &event
            )
        );
        result[R_STORE + 1] = (uint32_t)(kmx_time() - start);
    }
    kmx_result(0, result, RESULT_WORDS);
    return 0;
}
