/* Three patterns on the four cores with kmx_parallel, each run with one partition and
 * with four, timed: a map, a reduction, a two-pass prefix sum. See README.md. */
#include <stdint.h>

#include "kmx.h"
#include "vmap.h"

/* Regions in the host's order; words; result words (timings, then 64-bit sums). */
enum {
    REGION_INPUT,
    REGION_MAP_1,
    REGION_MAP_4,
    REGION_MAP_QUARTERS,
    REGION_MAP_ALIASED,
    REGION_VMAP_1,
    REGION_VMAP_4,
    REGION_HEAVY_1,
    REGION_HEAVY_4,
    REGION_SEQUENCE,
    REGION_PREFIX_1,
    REGION_PREFIX_4,
    REGION_COUNTERS,
    REGION_COUNT
};

enum { WORD_ARRAY_LENGTH, WORD_REPEATS, WORD_ROUNDS, WORD_COUNT };

enum {
    TIME_MAP_1,
    TIME_MAP_4,
    TIME_MAP_QUARTERS,
    TIME_MAP_ALIASED,
    TIME_VMAP_1,
    TIME_VMAP_4,
    TIME_HEAVY_1,
    TIME_HEAVY_4,
    TIME_SUM_1,
    TIME_SUM_4,
    TIME_SUM_PACKED,
    TIME_SUM_PADDED,
    TIME_SEQUENCE,
    TIME_PREFIX_1,
    TIME_PREFIX_4,
    TIME_INVALIDATE_1,
    TIME_INVALIDATE_4,
    TIME_FLUSH_1,
    TIME_FLUSH_4,
    TIME_PARALLEL,
    RESULT_SUMS,
    RESULT_WORDS = RESULT_SUMS + 8
};

#define CHUNK 1024u /* words of one partition: 4 KiB, whole 64-byte lines */
#define CHUNKS_MAX 4096u
#define PARALLEL_CALLS 1000u
/* The cluster's L3 is one 128 KiB way: the aliased map writes this far into its
 * region, which then starts a whole number of ways from the input. */
#define HALF_WAY_WORDS ((64u << 10) / 4u)

/* Per-core results, one cache line each (globals need .data and an initialiser). */
static uint64_t core_sums[4][8] __attribute__((section(".data"), aligned(64))) = {{1}};
/* The prefix sum's per-chunk sums, then each chunk's base. */
static uint32_t chunk_sums[CHUNKS_MAX] __attribute__((section(".data"))) = {1};

typedef struct {
    const uint32_t* input;
    uint32_t* output;
    uint64_t words;
    uint32_t partitions;         /* 1 or 4: a core takes every `partitions`-th chunk */
    uint32_t rounds;             /* mix() applied this many times per word */
    uint32_t quarters;           /* 1: a core takes its quarter of the words instead */
    uint32_t stride;             /* words of uint64_t between two cores' counters */
    volatile uint64_t* counters; /* the reduction's counters in a region, or 0 */
} job_t;

static inline uint64_t chunks(const job_t* job) {
    return (job->words + CHUNK - 1) / CHUNK;
}

/* A core's first chunk; its next ones are `partitions` apart. None past `partitions`.
 */
static inline uint64_t first(const job_t* job, int core) {
    return (uint32_t)core < job->partitions ? (uint64_t)core : chunks(job);
}

static inline uint64_t end(const job_t* job, uint64_t chunk) {
    return (chunk + 1) * CHUNK < job->words ? (chunk + 1) * CHUNK : job->words;
}

static inline uint32_t mix(uint32_t x) {
    return (x ^ (x >> 15)) * 2246822519u + 1u;
}

/* (a) map: output[word] = mix(input[word]), `rounds` times. The job's fields are read
 * once: a uint32_t store may alias job->rounds, which would be re-read every word. */
static void map_range(const job_t* job, uint64_t first_word, uint64_t end_word) {
    const uint32_t* input = job->input;
    uint32_t* output = job->output;
    const uint32_t rounds = job->rounds;
    for (uint64_t word = first_word; word < end_word; word++) {
        uint32_t x = input[word];
        for (uint32_t round = 0; round < rounds; round++) {
            x = mix(x);
        }
        output[word] = x;
    }
}

static void map_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    if (job->quarters) {
        const uint64_t quarter = (uint64_t)core;
        map_range(job, job->words * quarter / 4, job->words * (quarter + 1) / 4);
        return;
    }
    for (
        uint64_t chunk = first(job, core); chunk < chunks(job); chunk += job->partitions
    ) {
        map_range(job, chunk * CHUNK, end(job, chunk));
    }
}

/* The same map (one round) with the vector unit, vmap.c. */
static void vmap_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    for (
        uint64_t chunk = first(job, core); chunk < chunks(job); chunk += job->partitions
    ) {
        vmap(
            job->output + chunk * CHUNK,
            job->input + chunk * CHUNK,
            end(job, chunk) - chunk * CHUNK
        );
    }
}

/* (b) reduction: the core's sum in a register, written once to its line; or, with
 * `counters`, added word by word to a counter in a region. */
static void sum_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    uint64_t total = 0;
    volatile uint64_t*
        counter = (job->counters ? job->counters + (uint64_t)core * job->stride : 0);
    for (
        uint64_t chunk = first(job, core); chunk < chunks(job); chunk += job->partitions
    ) {
        for (uint64_t word = chunk * CHUNK; word < end(job, chunk); word++) {
            if (counter) {
                *counter += job->input[word];
            } else {
                total += job->input[word];
            }
        }
    }
    if ((uint32_t)core < job->partitions) {
        core_sums[core][0] = total;
    }
}

/* (c) prefix, pass 1: each chunk's sum. */
static void sum_chunks(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    /* my chunks of CHUNK words: core, core + partitions, ... (partitions: the cores in
     * use) */
    for (
        uint64_t chunk = first(job, core); chunk < chunks(job); chunk += job->partitions
    ) {
        uint32_t total = 0;
        for (uint64_t word = chunk * CHUNK; word < end(job, chunk); word++) {
            total += job->input[word];
        }
        chunk_sums[chunk] = total;
    }
}

/* Pass 2: output[word] = the chunk's base + the chunk's words before it. */
static void chunk_prefix(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    for (
        uint64_t chunk = first(job, core); chunk < chunks(job); chunk += job->partitions
    ) {
        uint32_t running_total = chunk_sums[chunk];
        for (uint64_t word = chunk * CHUNK; word < end(job, chunk); word++) {
            job->output[word] = running_total;
            running_total += job->input[word];
        }
    }
}

static void nothing(int core, void* arg) {
    (void)core;
    (void)arg;
}

/* Runs `function` on core 0 alone or on all four; KMX_OK, or kmx_parallel's error. */
static int run_on_cores(void (*function)(int, void*), job_t* job) {
    if (job->partitions == 1) {
        function(0, job);
        return KMX_OK;
    }
    return kmx_parallel(function, job);
}

/* The two-pass exclusive prefix sum (mod 2^32); between the passes core 0 turns the
 * chunk sums into bases. */
static int prefix(job_t* job) {
    const int error = run_on_cores(sum_chunks, job);
    uint32_t base = 0;
    for (uint64_t chunk = 0; chunk < chunks(job); chunk++) {
        const uint32_t chunk_sum = chunk_sums[chunk];
        chunk_sums[chunk] = base;
        base += chunk_sum;
    }
    return error != KMX_OK ? error : run_on_cores(chunk_prefix, job);
}

/* The one-pass prefix sum on core 0: the baseline two passes are measured against. */
static void prefix_sequence(job_t* job) {
    uint32_t running_total = 0;
    for (uint64_t word = 0; word < job->words; word++) {
        job->output[word] = running_total;
        running_total += job->input[word];
    }
}

/* Region `swept_region` swept by core 0 alone (the others idle): one core's flush or
 * invalidate. */
static int swept_region __attribute__((section(".data"))) = -1;

static void sweep_alone(int core, void* arg) {
    (void)arg;
    if (core == 0) {
        kmx_invalidate(swept_region); /* inside kmx_parallel: this core alone */
    }
}

typedef enum { MAP, VMAP, SUM, PREFIX, SEQUENCE } kind_t;

/* Best of `repeats` runs of one pattern, each from a cold cache (input and output
 * swept first); the output is flushed after. KMX_OK, or kmx_parallel's error. */
static int timed(
    kind_t kind, job_t* job, int output_region, uint32_t repeats, uint32_t* best
) {
    *best = 0xFFFFFFFFu;
    for (uint32_t repeat = 0; repeat < repeats; repeat++) {
        kmx_invalidate(REGION_INPUT);
        if (output_region >= 0) {
            kmx_invalidate(output_region);
        }
        const uint64_t start = kmx_time();
        int error = KMX_OK;
        if (kind == MAP || kind == VMAP) {
            error = run_on_cores(kind == MAP ? map_part : vmap_part, job);
        } else if (kind == SUM) {
            error = run_on_cores(sum_part, job);
        } else if (kind == PREFIX) {
            error = prefix(job);
        } else {
            prefix_sequence(job);
        }
        const uint32_t ticks = (uint32_t)(kmx_time() - start);
        if (error != KMX_OK) {
            return error;
        }
        *best = ticks < *best ? ticks : *best;
    }
    if (output_region >= 0) {
        kmx_flush(output_region);
    }
    return KMX_OK;
}

static void store_sum(uint32_t* result, uint64_t sum) {
    result[0] = (uint32_t)sum;
    result[1] = (uint32_t)(sum >> 32);
}

/* The maps: output region, result word, cores, heavy (mix `rounds` times), quarters,
 * the output's offset in words. */
static const uint32_t MAPS[8][6] = {
    {REGION_MAP_1, TIME_MAP_1, 1, 0, 0, 0},
    {REGION_MAP_4, TIME_MAP_4, 4, 0, 0, 0},
    {REGION_MAP_QUARTERS, TIME_MAP_QUARTERS, 4, 0, 1, 0},
    {REGION_MAP_ALIASED, TIME_MAP_ALIASED, 1, 0, 0, HALF_WAY_WORDS},
    {REGION_VMAP_1, TIME_VMAP_1, 1, 0, 0, 0},
    {REGION_VMAP_4, TIME_VMAP_4, 4, 0, 0, 0},
    {REGION_HEAVY_1, TIME_HEAVY_1, 1, 1, 0, 0},
    {REGION_HEAVY_4, TIME_HEAVY_4, 4, 1, 0, 0},
};

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();
    /* every word is set below (a zeroing loop may call memset) */
    uint32_t result_words[RESULT_WORDS];
    if (
        arguments->nwords < WORD_COUNT || arguments->nregions != REGION_COUNT ||
        arguments->words[WORD_REPEATS] == 0 ||
        arguments->words[WORD_ARRAY_LENGTH] == 0 ||
        arguments->words[WORD_ARRAY_LENGTH] > CHUNKS_MAX * CHUNK
    ) {
        return 1;
    }
    const uint64_t array_words = arguments->words[WORD_ARRAY_LENGTH];
    const uint32_t repeats = arguments->words[WORD_REPEATS];
    void* regions[REGION_COUNT];
    for (int region = 0; region < REGION_COUNT; region++) {
        uint64_t bytes;
        regions[region] = kmx_region(region, &bytes);
        uint64_t needed = (
            array_words * 4 + (region == REGION_MAP_ALIASED ? HALF_WAY_WORDS * 4 : 0)
        );
        needed = region == REGION_COUNTERS ? 4 * 64 : needed;
        if (regions[region] == 0 || bytes < needed) {
            return 1;
        }
    }
    job_t job = {(const uint32_t*)regions[REGION_INPUT], 0, array_words, 1, 1, 0, 0, 0};
    for (int map_index = 0; map_index < 8; map_index++) {
        kmx_progress(1 + map_index);
        job.output = (uint32_t*)regions[MAPS[map_index][0]] + MAPS[map_index][5];
        job.partitions = MAPS[map_index][2];
        job.rounds = MAPS[map_index][3] ? arguments->words[WORD_ROUNDS] : 1;
        job.quarters = MAPS[map_index][4];
        const kind_t kind = map_index == 4 || map_index == 5 ? VMAP : MAP;
        if (timed(
            kind,
            &job,
            (int)MAPS[map_index][0],
            repeats,
            &result_words[MAPS[map_index][1]]
        )) {
            return 2;
        }
    }
    /* reductions: in a register (one core, four), then in counters (packed, padded) */
    job.output = 0;
    job.quarters = 0;
    volatile uint64_t* counters = (volatile uint64_t*)regions[REGION_COUNTERS];
    for (int sum_index = 0; sum_index < 4; sum_index++) {
        kmx_progress(9 + sum_index);
        job.partitions = sum_index == 0 ? 1 : 4;
        job.counters = sum_index < 2 ? 0 : counters;
        job.stride = sum_index == 2 ? 1 : 8;
        for (uint32_t i = 0; i < 32; i++) {
            counters[i] = 0;
        }
        if (timed(SUM, &job, -1, repeats, &result_words[TIME_SUM_1 + sum_index])) {
            return 2;
        }
        uint64_t total = 0;
        for (uint32_t core_index = 0; core_index < job.partitions; core_index++) {
            total += (
                sum_index < 2
                    ? core_sums[core_index][0]
                    : counters[core_index * job.stride]
            );
        }
        /* counters kept adding */
        store_sum(
            &result_words[RESULT_SUMS + 2 * sum_index],
            sum_index < 2 ? total : total / repeats
        );
    }
    job.counters = 0;
    static const int prefix_outputs[3] = {
        REGION_SEQUENCE, REGION_PREFIX_1, REGION_PREFIX_4
    };
    for (int prefix_index = 0; prefix_index < 3; prefix_index++) {
        kmx_progress(13 + prefix_index);
        job.output = (uint32_t*)regions[prefix_outputs[prefix_index]];
        job.partitions = prefix_index == 2 ? 4 : 1;
        if (timed(
            prefix_index == 0 ? SEQUENCE : PREFIX,
            &job,
            prefix_outputs[prefix_index],
            repeats,
            &result_words[TIME_SEQUENCE + prefix_index]
        )) {
            return 2;
        }
    }
    /* flush and invalidate: a region just read (clean lines), then one just written
     * (dirty), swept by core 0 alone and by all four */
    kmx_progress(16);
    job.output = (uint32_t*)regions[REGION_MAP_4];
    job.partitions = 4;
    job.rounds = 1;
    for (int dirty = 0; dirty < 2; dirty++) {
        const int id = dirty == 0 ? REGION_INPUT : REGION_MAP_4;
        for (int x280_cores = 1; x280_cores <= 4; x280_cores += 3) {
            if (run_on_cores(dirty == 0 ? sum_part : map_part, &job) != KMX_OK) {
                return 2;
            }
            const uint64_t start = kmx_time();
            if (x280_cores == 4) {
                kmx_invalidate(id);
            } else {
                swept_region = id;
                if (kmx_parallel(sweep_alone, 0) != KMX_OK) {
                    return 2;
                }
            }
            result_words
                [
                    (dirty ? TIME_FLUSH_1 : TIME_INVALIDATE_1) + x280_cores / 4
                ] = (uint32_t)(kmx_time() - start);
        }
    }
    kmx_progress(17);
    const uint64_t start = kmx_time();
    for (uint32_t i = 0; i < PARALLEL_CALLS; i++) {
        if (kmx_parallel(nothing, 0) != KMX_OK) {
            return 2;
        }
    }
    result_words[TIME_PARALLEL] = (uint32_t)(kmx_time() - start);
    kmx_result(0, result_words, RESULT_WORDS);
    return 0;
}
