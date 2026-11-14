/* Per count of Tensix cores and way of fetching: region 0 zeroed, `repeats` calls
 * timed, its hash; then a store, and the sweep's time on four cores and on one. */
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
#define COUNTS_MAX 3u
/* the ways, each with its best call's ticks at R_WAY + 3 way and its hash after it */
#define WAY_CALL 0u   /* kmx_fetch: one sweep, the zeroed region's lines dirty */
#define WAY_CLEAN 1u  /* kmx_fetch_clean after a flush of the range: no sweep */
#define WAY_BEFORE 2u /* kmx_fetch then kmx_invalidate_at: the earlier call's work */
#define WAYS 3u
/* result words: per count i at 12 i, then the store's at 36 and the sweeps' at 38 */
/* the call's best event ticks, the dispatcher's submit to completion */
#define R_EVENT 0u
/* the first failure: status | event kind << 8 | aux << 16 | way << 24 */
#define R_STATUS 1u
#define R_WORST 2u  /* the call's worst ticks */
#define R_WAY 3u    /* + 3 way: best ticks, then the hash's two words */
#define R_STORE 36u /* status, call ticks */
/* kmx_flush_at of the range: ticks on four cores, on core 0 alone */
#define R_SWEEP 38u
#define RESULT_WORDS 40u

/* The region's hash, as the host's numpy computes it. */
static uint64_t hash(const volatile uint64_t* words, uint64_t count) {
    uint64_t sum = 0;
    for (uint64_t j = 0; j < count; j++) {
        sum += words[j] * (2 * j + 1);
    }
    return sum;
}

/* kmx_parallel's function: core 0 sweeps the whole range, alone inside the call. */
static void sweep_alone(int core, void* arg) {
    if (core == 0) {
        kmx_flush_at(0, 0, *(const uint64_t*)arg);
    }
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
    uint64_t bytes = words[W_BYTES];
    if (region == 0 || bytes > region_bytes || words[W_COUNTS] > COUNTS_MAX) {
        return 1;
    }
    const kmx_dram range = {words[W_ADDRESS], words[W_BANK], 0, 0};
    for (uint32_t i = 0; i < words[W_COUNTS]; i++) {
        const uint32_t count = words[W_COUNT + i];
        uint32_t* out = &result[12 * i];
        for (uint32_t way = 0; way < WAYS && out[R_STATUS] == 0; way++) {
            uint32_t* best = &out[R_WAY + 3 * way];
            kmx_progress(1 + WAYS * i + way);
            for (uint64_t j = 0; j < bytes / 8; j++) {
                region[j] = 0;
            }
            if (way == WAY_CLEAN) {
                kmx_flush_at(0, 0, bytes);
            }
            best[0] = 0xFFFFFFFFu;
            for (uint32_t r = 0; r < words[W_REPEATS]; r++) {
                kmx_event event = {0, 0, 0, 0};
                const uint64_t start = kmx_time();
                const uint64_t deadline = start + KMX_TICKS_PER_SECOND;
                const int status = (
                    way == WAY_CLEAN
                        ? kmx_fetch_clean(
                            0, 0, &range, bytes, 0, count, deadline, &event
                        )
                        : kmx_fetch(0, 0, &range, bytes, 0, count, deadline, &event)
                );
                if (way == WAY_BEFORE && status == KMX_OK) {
                    kmx_invalidate_at(0, 0, bytes);
                }
                const uint32_t ticks = (uint32_t)(kmx_time() - start);
                if (status != KMX_OK) {
                    out[R_STATUS] = (
                        (uint32_t)status | event.kind << 8 | event.auxiliary << 16 |
                        way << 24
                    );
                    break;
                }
                if (ticks < best[0]) {
                    best[0] = ticks;
                    out[R_EVENT] = way == WAY_CALL ? event.ticks : out[R_EVENT];
                }
                if (way == WAY_CALL) {
                    out[R_WORST] = ticks > out[R_WORST] ? ticks : out[R_WORST];
                }
            }
            const uint64_t sum = hash(region, bytes / 8);
            best[1] = (uint32_t)sum;
            best[2] = (uint32_t)(sum >> 32);
        }
    }
    if (words[W_STORE] != 0) {
        kmx_progress(9 + WAYS * COUNTS_MAX);
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
    uint64_t start = kmx_time();
    kmx_flush_at(0, 0, bytes);
    result[R_SWEEP] = (uint32_t)(kmx_time() - start);
    start = kmx_time();
    if (kmx_parallel(sweep_alone, &bytes) == KMX_OK) {
        result[R_SWEEP + 1] = (uint32_t)(kmx_time() - start);
    }
    kmx_result(0, result, RESULT_WORDS);
    return 0;
}
