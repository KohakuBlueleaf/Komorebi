/* Fetch and store over islands: per size, kmx_fetch of a tensor of this island and of
 * the other into region 0 (best of `repeats`, hash), then kmx_store into each. */
#include <stdint.h>

#include "kmx.h"

/* words */
#define W_LOCAL 0u /* the tensor this island holds */
#define W_FAR 1u   /* the same bytes, held by the other island */
#define W_PAGE 2u  /* both tensors' aligned page bytes */
#define W_COUNT 3u /* Tensix cores a copy */
#define W_REPEATS 4u
#define W_SIZES 5u /* how many sizes follow, at most SIZES_MAX */
#define W_SIZE 6u  /* bytes of each size, then the store tensors of each: */
#define SIZES_MAX 2u
#define W_BACK 8u /* per size: the local store tensor, then the far one */
#define BANK_TENSOR 0xFFu
/* result words: per size at 10 i */
/* the first failure: status | event kind << 8 | aux << 16 | way << 24 */
#define R_STATUS 0u
#define R_LOCAL 1u /* best ticks, then the hash's two words */
#define R_FAR 4u   /* best ticks, then the hash's two words */
#define R_STORE_LOCAL 7u
#define R_STORE_FAR 8u
#define RESULT_WORDS 20u

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
    if (region == 0 || words[W_SIZES] > SIZES_MAX) {
        return 1;
    }
    const uint32_t count = words[W_COUNT];
    for (uint32_t i = 0; i < words[W_SIZES]; i++) {
        const uint64_t bytes = words[W_SIZE + i];
        uint32_t* out = &result[10 * i];
        if (bytes > region_bytes) {
            return 1;
        }
        for (uint32_t way = 0; way < 2 && out[R_STATUS] == 0; way++) {
            const kmx_dram from = {
                words[way == 0 ? W_LOCAL : W_FAR], BANK_TENSOR, words[W_PAGE], 0
            };
            uint32_t* best = &out[way == 0 ? R_LOCAL : R_FAR];
            best[0] = 0xFFFFFFFFu;
            kmx_progress(1 + 4 * i + way);
            for (uint32_t r = 0; r < words[W_REPEATS]; r++) {
                for (uint64_t j = 0; j < bytes / 8; j++) {
                    region[j] = 0;
                }
                kmx_event event = {0, 0, 0, 0};
                const uint64_t start = kmx_time();
                const int status = (
                    kmx_fetch(
                        0, 0, &from, bytes, 0, count, start + KMX_TICKS_PER_SECOND, &event
                    )
                );
                const uint32_t ticks = (uint32_t)(kmx_time() - start);
                if (status != KMX_OK) {
                    out[R_STATUS] = (
                        (uint32_t)status | event.kind << 8 | event.auxiliary << 16 |
                        way << 24
                    );
                    break;
                }
                best[0] = ticks < best[0] ? ticks : best[0];
            }
            const uint64_t sum = hash(region, bytes / 8);
            best[1] = (uint32_t)sum;
            best[2] = (uint32_t)(sum >> 32);
        }
        /* the region holds the far fetch's bytes: stored into each island's tensor */
        for (uint32_t way = 0; way < 2 && out[R_STATUS] == 0; way++) {
            const kmx_dram to = {
                words[W_BACK + 2 * i + way], BANK_TENSOR, words[W_PAGE], 0
            };
            kmx_event event = {0, 0, 0, 0};
            kmx_progress(3 + 4 * i + way);
            const uint64_t start = kmx_time();
            const int status = (
                kmx_store(
                    0, 0, &to, bytes, 0, count, start + KMX_TICKS_PER_SECOND, &event
                )
            );
            out[R_STORE_LOCAL + way] = (uint32_t)(kmx_time() - start);
            if (status != KMX_OK) {
                out[R_STATUS] = (
                    (uint32_t)status | event.kind << 8 | event.auxiliary << 16 |
                    (2 + way) << 24
                );
            }
        }
    }
    kmx_result(0, result, RESULT_WORDS);
    return 0;
}
