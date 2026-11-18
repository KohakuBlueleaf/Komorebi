/* A map, a heavy map and a two-pass prefix sum on 1, 4, 8 and 12 cores: one image on
 * every cluster, the lead's kmx_parallel_all, data pushed across banks (README.md). */
#include <stdint.h>

#include "kmx.h"

/* Regions in the host's order (the same on every cluster); words; result words. */
enum {
    REGION_INPUT,
    REGION_REFERENCE_MAP,
    REGION_REFERENCE_HEAVY,
    REGION_REFERENCE_PREFIX,
    REGION_MAP,
    REGION_HEAVY,
    REGION_PREFIX,
    REGION_SUMS,
    REGION_COUNT
};

enum { WORD_ARRAY_LENGTH, WORD_REPEATS, WORD_ROUNDS, WORD_COUNT };

/* Per pattern (map, heavy map, prefix) and cores (1, 4, 8, 12): times, then words that
 * differ from the 1-core output; then the input push's times. */
enum { RESULT_TIMES = 0, RESULT_DIFFERS = 12, RESULT_PUSHES = 24, RESULT_WORDS = 28 };

#define CHUNK 1024u /* words of one partition: 4 KiB */
#define CHUNKS_MAX 4096u
/* words: a chunk's sum has a line of its own, so each block's sums push */
#define LINE 16u
#define CODE_BAD 1u
#define CODE_LATE 2u
#define CODE_PUSH 3u

/* What every core runs (the call's word: cores | phase << 8 | region << 16, the
 * region written or sent); REGION_HEAVY's map applies mix `rounds` times. */
enum { PHASE_MAP, PHASE_SUMS, PHASE_PREFIX, PHASE_SEND };

static const uint32_t X280_CORES[4] = {1, 4, 8, 12};
/* Set by a core whose push failed (globals need .data and an initialiser). */
static volatile uint32_t push_failed __attribute__((section(".data"))) = 0;

static inline uint32_t mix(uint32_t x) {
    return (x ^ (x >> 15)) * 2246822519u + 1u;
}

static uint64_t words(void) {
    return kmx_args_get()->words[WORD_ARRAY_LENGTH];
}

static uint64_t chunks(void) {
    return (words() + CHUNK - 1) / CHUNK;
}

/* The clusters in use with `cores` (the lowest first, the lead), and `cluster`'s
 * index among them; -1 when it takes no part. */
static int index_of(uint32_t x280_cores, int cluster, uint32_t* cluster_count) {
    const uint32_t mask = kmx_clusters();
    *cluster_count = x280_cores <= 4 ? 1 : x280_cores / 4;
    uint32_t index = 0;
    for (int candidate = 1; candidate < 4; candidate++) {
        if (mask >> candidate & 1u) {
            if (candidate == cluster) {
                return index < *cluster_count ? (int)index : -1;
            }
            index++;
        }
    }
    return -1;
}

/* Cluster `index` of `cluster_count`: chunks [first_chunk, end_chunk), whole groups
 * each. */
static void block(
    uint32_t index, uint32_t cluster_count, uint64_t* first_chunk, uint64_t* end_chunk
) {
    *first_chunk = chunks() * index / cluster_count;
    *end_chunk = chunks() * (index + 1) / cluster_count;
    *end_chunk = *end_chunk < chunks() ? *end_chunk : chunks();
    *first_chunk = *first_chunk < *end_chunk ? *first_chunk : *end_chunk;
}

/* Bytes [offset, offset + length) of 32-bit words [first_word, end_word), whole
 * lines (regions are padded). */
static void bytes_of(
    uint64_t first_word, uint64_t end_word, uint64_t* offset, uint64_t* length
) {
    *offset = first_word * 4;
    *length = ((end_word - first_word) * 4 + 63) / 64 * 64;
}

static uint64_t end_of(uint64_t chunk) {
    return (chunk + 1) * CHUNK < words() ? (chunk + 1) * CHUNK : words();
}

/* One core's partition of a phase: chunks first_chunk + core, first_chunk + core +
 * partitions, ... */
static void step(int cluster, int core, uint64_t arg) {
    const uint32_t x280_cores = arg & 0xFFu, phase = (arg >> 8) & 0xFFu;
    uint32_t cluster_count;
    const int index = index_of(x280_cores, cluster, &cluster_count);
    const uint32_t partitions = x280_cores == 1 ? 1 : 4;
    if (index < 0 || (uint32_t)core >= partitions) {
        return;
    }
    uint64_t first_chunk, end_chunk;
    block((uint32_t)index, cluster_count, &first_chunk, &end_chunk);
    const uint32_t* input = (const uint32_t*)kmx_region(REGION_INPUT, 0);
    uint32_t* sums = (uint32_t*)kmx_region(REGION_SUMS, 0);
    if (phase == PHASE_SEND) {
        /* not the lead: its quarter of the block's range of a region, to the lead */
        const int region = (int)((arg >> 16) & 0xFFu);
        const uint64_t unit = region == REGION_SUMS ? LINE : CHUNK;
        uint64_t offset, length;
        bytes_of(first_chunk * unit, end_chunk * unit, &offset, &length);
        const uint64_t lines = length / 64;
        const uint64_t lines_per_core = (lines + 3) / 4;
        const uint64_t first_line = lines_per_core * core;
        if (index == 0 || first_line >= lines) {
            return;
        }
        offset += first_line * 64;
        const uint64_t remaining = lines - first_line;
        length = (remaining < lines_per_core ? remaining : lines_per_core) * 64;
        const uint8_t* source = (const uint8_t*)kmx_region(region, 0) + offset;
        if (kmx_push(kmx_lead(), region, offset, source, length) != KMX_OK) {
            push_failed = 1;
        }
        if (region == REGION_SUMS) {
            /* the lead pushes the bases back over them */
            kmx_flush_at(region, offset, length);
        }
        return;
    }
    if (index != 0 && phase == PHASE_PREFIX) {
        uint64_t offset, length;
        bytes_of(first_chunk * LINE, end_chunk * LINE, &offset, &length);
        kmx_invalidate_at(REGION_SUMS, offset, length); /* the bases the lead pushed */
    }
    const int region = (int)((arg >> 16) & 0xFFu);
    const uint32_t rounds = (
        region == REGION_HEAVY ? kmx_args_get()->words[WORD_ROUNDS] : 1
    );
    uint32_t* output = (uint32_t*)kmx_region(region, 0);
    for (
        uint64_t chunk = first_chunk + (uint64_t)core; chunk < end_chunk;
        chunk += partitions
    ) {
        const uint64_t first_word = chunk * CHUNK, end_word = end_of(chunk);
        if (index != 0 && phase != PHASE_PREFIX) {
            /* what the lead pushed */
            kmx_invalidate_at(
                REGION_INPUT, first_word * 4, (end_word - first_word) * 4
            );
        }
        if (phase == PHASE_MAP) {
            for (uint64_t word = first_word; word < end_word; word++) {
                uint32_t x = input[word];
                for (uint32_t round = 0; round < rounds; round++) {
                    x = mix(x);
                }
                output[word] = x;
            }
        } else if (phase == PHASE_SUMS) {
            uint32_t total = 0;
            for (uint64_t word = first_word; word < end_word; word++) {
                total += input[word];
            }
            sums[chunk * LINE] = total;
        } else {
            uint32_t running_total = sums[chunk * LINE];
            for (uint64_t word = first_word; word < end_word; word++) {
                output[word] = running_total;
                running_total += input[word];
            }
        }
    }
}

typedef struct {
    uint64_t arg;
} local_t;

static void local_step(int core, void* arg) {
    step(kmx_lead(), core, ((const local_t*)arg)->arg);
}

/* One phase on the cores in use: core 0 alone, the lead's four, or every cluster. */
static int run_phase(uint32_t x280_cores, uint32_t phase, int region) {
    const uint64_t arg = x280_cores | phase << 8 | (uint64_t)region << 16;
    if (x280_cores == 1) {
        step(kmx_lead(), 0, arg);
        return KMX_OK;
    }
    if (x280_cores == 4) {
        local_t local = {arg};
        return kmx_parallel(local_step, &local);
    }
    return kmx_parallel_all(step, arg);
}

/* The lead's pushes of `region`'s blocks (in `unit` words per chunk) to the others in
 * use, or its takes of what they sent back. */
static int blocks(uint32_t x280_cores, int region, uint64_t unit, int take) {
    uint32_t cluster_count;
    (void)index_of(x280_cores, kmx_lead(), &cluster_count);
    const uint32_t mask = kmx_clusters();
    uint32_t index = 0;
    for (int cluster = 1; cluster < 4; cluster++) {
        if (!(mask >> cluster & 1u)) {
            continue;
        }
        uint64_t first_chunk, end_chunk, offset, length;
        block(index, cluster_count, &first_chunk, &end_chunk);
        bytes_of(first_chunk * unit, end_chunk * unit, &offset, &length);
        if (index > 0 && index < cluster_count && length != 0) {
            if (take) {
                kmx_invalidate_at(region, offset, length);
            } else {
                const uint8_t* source = (const uint8_t*)kmx_region(region, 0) + offset;
                if (kmx_push(cluster, region, offset, source, length) != KMX_OK) {
                    return KMX_BAD;
                }
            }
        }
        index++;
    }
    return KMX_OK;
}

/* A map into `region` on `cores`: the input out, the map, the outputs back. */
static int map(uint32_t x280_cores, int region, uint32_t* push_ticks) {
    const uint64_t start = kmx_time();
    int error = blocks(x280_cores, REGION_INPUT, CHUNK, 0);
    *push_ticks = (uint32_t)(kmx_time() - start);
    error = error ? error : run_phase(x280_cores, PHASE_MAP, region);
    error = (
        error
            ? error
            : (x280_cores > 4 ? run_phase(x280_cores, PHASE_SEND, region) : KMX_OK)
    );
    return error ? error : blocks(x280_cores, region, CHUNK, 1);
}

/* The two-pass prefix sum: the chunk sums to the lead, the bases back out. */
static int prefix(uint32_t x280_cores) {
    const int all_clusters = x280_cores > 4;
    int error = blocks(x280_cores, REGION_INPUT, CHUNK, 0);
    error = error ? error : run_phase(x280_cores, PHASE_SUMS, 0);
    error = (
        error
            ? error
            : (all_clusters ? run_phase(x280_cores, PHASE_SEND, REGION_SUMS) : KMX_OK)
    );
    error = error ? error : blocks(x280_cores, REGION_SUMS, LINE, 1);
    if (error) {
        return error;
    }
    uint32_t* sums = (uint32_t*)kmx_region(REGION_SUMS, 0);
    uint32_t base = 0;
    for (uint64_t chunk = 0; chunk < chunks(); chunk++) {
        const uint32_t chunk_sum = sums[chunk * LINE];
        sums[chunk * LINE] = base;
        base += chunk_sum;
    }
    error = blocks(x280_cores, REGION_SUMS, LINE, 0);
    error = error ? error : run_phase(x280_cores, PHASE_PREFIX, REGION_PREFIX);
    error = (
        error
            ? error
            : (all_clusters ? run_phase(x280_cores, PHASE_SEND, REGION_PREFIX) : KMX_OK)
    );
    return error ? error : blocks(x280_cores, REGION_PREFIX, CHUNK, 1);
}

static uint32_t differ(const uint32_t* left, const uint32_t* right) {
    uint32_t count = 0;
    for (uint64_t word = 0; word < words(); word++) {
        count += left[word] != right[word];
    }
    return count;
}

static void copy(uint32_t* destination, const uint32_t* source) {
    for (uint64_t word = 0; word < words(); word++) {
        destination[word] = source[word];
    }
}

int kmx_main(int core) {
    (void)core;
    if (kmx_cluster() != kmx_lead()) {
        const int error = kmx_serve();
        return error != KMX_OK ? (int)CODE_LATE : push_failed ? (int)CODE_PUSH : 0;
    }
    const kmx_args* arguments = kmx_args_get();
    uint32_t result_words[RESULT_WORDS]; /* every word is set below */
    if (
        arguments->nwords < WORD_COUNT || arguments->nregions != REGION_COUNT ||
        arguments->words[WORD_REPEATS] == 0 || words() == 0 || chunks() > CHUNKS_MAX
    ) {
        return CODE_BAD;
    }
    for (int region = 0; region < REGION_COUNT; region++) {
        uint64_t bytes;
        const uint64_t needed = (
            region == REGION_SUMS ? CHUNKS_MAX * 64 : (words() * 4 + 63) / 64 * 64
        );
        if (kmx_region(region, &bytes) == 0 || bytes < needed) {
            return CODE_BAD;
        }
    }
    uint32_t* regions[REGION_COUNT];
    for (int region = 0; region < REGION_COUNT; region++) {
        regions[region] = (uint32_t*)kmx_region(region, 0);
    }
    static const int OUTPUTS[3] = {REGION_MAP, REGION_HEAVY, REGION_PREFIX};
    for (int cores_index = 0; cores_index < 4; cores_index++) {
        for (int pattern = 0; pattern < 3; pattern++) {
            uint32_t best = 0xFFFFFFFFu;
            for (
                uint32_t repeat = 0; repeat < arguments->words[WORD_REPEATS]; repeat++
            ) {
                kmx_progress(1 + 3 * cores_index + pattern);
                kmx_invalidate(REGION_INPUT); /* every run from a cold cache */
                kmx_invalidate(OUTPUTS[pattern]);
                kmx_flush(REGION_SUMS); /* no dirty line where the others' sums land */
                const uint64_t start = kmx_time();
                int error;
                if (pattern < 2) {
                    error = (
                        map(
                            X280_CORES[cores_index],
                            OUTPUTS[pattern],
                            &result_words[RESULT_PUSHES + cores_index]
                        )
                    );
                } else {
                    error = prefix(X280_CORES[cores_index]);
                }
                const uint32_t ticks = (uint32_t)(kmx_time() - start);
                if (error != KMX_OK) {
                    return error == KMX_BAD ? (int)CODE_PUSH : (int)CODE_LATE;
                }
                best = ticks < best ? ticks : best;
            }
            result_words[RESULT_TIMES + 4 * pattern + cores_index] = best;
            if (cores_index == 0) {
                /* the 1-core output: the reference */
                copy(regions[OUTPUTS[pattern] - 3], regions[OUTPUTS[pattern]]);
            }
            result_words[RESULT_DIFFERS + 4 * pattern + cores_index] = (
                differ(regions[OUTPUTS[pattern]], regions[OUTPUTS[pattern] - 3])
            );
        }
    }
    kmx_result(push_failed ? CODE_PUSH : 0, result_words, RESULT_WORDS);
    return 0;
}
