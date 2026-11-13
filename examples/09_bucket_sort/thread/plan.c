/* Bucket sort, X280 side: the THREAD, one image on each work cluster taking part. Core
 * 0 of cluster 1 drives the whole program; the plan runs on 1, 4 or 12 cores. */
#include <stdint.h>

#include "bucket_sort.h"
#include "kmx.h"

/* The regions, in the host's order. Each cluster has its own copy, in its own bank. */
enum { REGION_COUNTS, REGION_PLAN, REGION_TOTALS, REGION_PEERS, REGION_COUNT };

/* What the planning cores do in one call (step() below). */
enum { PHASE_HELLO, PHASE_TOTALS, PHASE_SEND, PHASE_PLAN };

/* The thread's codes (0: done). */
enum { CODE_BAD = 1, CODE_PARALLEL, CODE_PUSH, CODE_RING, CODE_COMMAND };

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command: longer than its deadline */
#define LINE 64u                        /* bytes: one cache line, one chunk of a row */

/* Set by a core whose push failed (globals need .data and an initialiser). */
static volatile uint32_t push_failed __attribute__((section(".data"))) = 0;

static uint32_t word(int index) {
    return kmx_args_get()->words[index];
}

static uint32_t buckets(void) {
    return 1u << word(BS_WORD_LOG2_BUCKETS);
}

static uint32_t chunks(void) {
    return buckets() / BS_CHUNK;
}

static uint32_t blocks(void) {
    return word(BS_WORD_X280_CORES) == 12 ? 3u : 1u;
}

/* Bytes from one table row to the next: a counter of 4 bytes per bucket. */
static uint64_t row_bytes(void) {
    return (uint64_t)buckets() * 4;
}

/* Where the totals start in their region: half a row away, in the cluster's cache of
 * one 128 KiB way, from every row of the counts and plan beside them (README.md). */
static uint64_t skew(void) {
    const uint64_t bytes_per_row = row_bytes();
    const uintptr_t gap = (
        (uintptr_t)kmx_region(REGION_TOTALS, 0) -
        (uintptr_t)kmx_region(REGION_COUNTS, 0)
    );
    return (
        (bytes_per_row / 2 + bytes_per_row - gap % bytes_per_row) % bytes_per_row /
        LINE * LINE
    );
}

static uint32_t* totals_of(void) {
    return (uint32_t*)((uint8_t*)kmx_region(REGION_TOTALS, 0) + skew());
}

/* Block 0 is planned by cluster 1 (the lead), block 1 by the next cluster running this
 * image, and so on; these two functions turn one into the other. */
static uint32_t block_of(int cluster) {
    uint32_t block = 0;
    for (int candidate = 1; candidate < cluster; candidate++) {
        block += (kmx_clusters() >> candidate) & 1u;
    }
    return block;
}

static int cluster_of(uint32_t block) {
    for (int cluster = 1; cluster < 4; cluster++) {
        if (((kmx_clusters() >> cluster) & 1u) && block_of(cluster) == block) {
            return cluster;
        }
    }
    return 0;
}

/* Step 2, on every planning core: `arg` is the phase | cores per cluster << 8. Each
 * cluster plans its block of buckets, each of its cores a quarter of that block. */
static void step(int cluster, int core, uint64_t arg) {
    const uint32_t phase = arg & 0xFFu, partitions = (arg >> 8) & 0xFFu;
    if ((uint32_t)core >= partitions) {
        return;
    }
    const uint32_t block = block_of(cluster), bucket_count = buckets();
    const uint32_t rows = word(BS_WORD_CORES);
    const uint64_t bytes_per_row = row_bytes();
    uint32_t first_chunk, end_chunk;
    bs_block(block, blocks(), chunks(), &first_chunk, &end_chunk);
    /* This cluster's own copy of each region, in its own bank, read through its cache.
     */
    const uint32_t* counts = (const uint32_t*)kmx_region(REGION_COUNTS, 0);
    uint32_t* plan = (uint32_t*)kmx_region(REGION_PLAN, 0);
    uint32_t* totals = totals_of();
    if (phase == PHASE_HELLO || phase == PHASE_SEND) {
        /* One core per cluster other than the lead pushes a line to the lead. */
        if (block == 0 || core != 0) {
            return;
        }
        if (phase == PHASE_HELLO) {
            /* Step 0: the lead cannot see my regions' DRAM addresses; I push them
             * into my line of its `peers` region, for the commands it builds. */
            uint32_t* line = (uint32_t*)kmx_region(REGION_PEERS, 0) + 16 * cluster;
            line[0] = (uint32_t)kmx_args_get()->region[REGION_COUNTS].offset;
            line[1] = (uint32_t)kmx_args_get()->region[REGION_PLAN].offset;
            if (
                kmx_push(kmx_lead(), REGION_PEERS, LINE * cluster, line, LINE) != KMX_OK
            ) {
                push_failed = 1;
            }
            return;
        }
        /* Step 2b: my block's bucket totals to the lead's bank (it cannot read mine),
         * then flush them: the lead pushes the starts over these very lines. */
        if (end_chunk > first_chunk) {
            const uint64_t offset = skew() + (uint64_t)first_chunk * LINE;
            const uint64_t length = (uint64_t)(end_chunk - first_chunk) * LINE;
            if (kmx_push(
                kmx_lead(),
                REGION_TOTALS,
                offset,
                totals + first_chunk * BS_CHUNK,
                length
            ) != KMX_OK) {
                push_failed = 1;
            }
            kmx_flush_at(REGION_TOTALS, offset, length);
        }
        return;
    }
    /* My part of the block: a run of whole chunks, so no two cores share a line. */
    uint32_t core_first_chunk, core_end_chunk;
    bs_block(
        (uint32_t)core,
        partitions,
        end_chunk - first_chunk,
        &core_first_chunk,
        &core_end_chunk
    );
    const uint32_t first_bucket = (first_chunk + core_first_chunk) * BS_CHUNK;
    const uint32_t end_bucket = (first_chunk + core_end_chunk) * BS_CHUNK;
    const uint64_t offset = (uint64_t)first_bucket * 4;
    const uint64_t length = (uint64_t)(end_bucket - first_bucket) * 4;
    if (length == 0) {
        return;
    }
    if (phase == PHASE_TOTALS) {
        /* Step 2a: my buckets' totals, row after row (a row is read as one stream).
         * The cores wrote the rows over the NoC, past this cache: invalidate first. */
        for (uint32_t bucket = first_bucket; bucket < end_bucket; bucket++) {
            totals[bucket] = 0;
        }
        for (uint32_t row = 0; row < rows; row++) {
            kmx_invalidate_at(REGION_COUNTS, row * bytes_per_row + offset, length);
            const uint32_t* input = counts + (uint64_t)row * bucket_count;
            for (uint32_t bucket = first_bucket; bucket < end_bucket; bucket++) {
                totals[bucket] += input[bucket];
            }
        }
        return;
    }
    /* Step 2d: from each bucket's start (from the lead), row after row: a core's first
     * key of a bucket goes after those of the cores before it; totals become ends. */
    if (block != 0) {
        /* the starts the lead pushed */
        kmx_invalidate_at(REGION_TOTALS, skew() + offset, length);
    }
    for (uint32_t row = 0; row < rows; row++) {
        const uint32_t* input = counts + (uint64_t)row * bucket_count;
        uint32_t* output = plan + (uint64_t)row * bucket_count;
        for (uint32_t bucket = first_bucket; bucket < end_bucket; bucket++) {
            output[bucket] = totals[bucket];
            totals[bucket] += input[bucket];
        }
        /* The cores read the plan over the NoC, from DRAM: write the lines back. */
        kmx_flush_at(REGION_PLAN, row * bytes_per_row + offset, length);
    }
}

typedef struct {
    uint64_t arg;
} local_t;

static void local_step(int core, void* arg) {
    step(kmx_lead(), core, ((const local_t*)arg)->arg);
}

/* A phase of step 2 on the cores asked for: core 0 alone, cluster 1's four
 * (kmx_parallel), or all twelve (clusters 1 to 3) (kmx_parallel_all). All timed. */
static int run_phase(uint32_t phase) {
    const uint32_t x280_cores = word(BS_WORD_X280_CORES);
    const uint64_t arg = phase | (uint64_t)(x280_cores == 1 ? 1u : 4u) << 8;
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

/* Step 2c, on the lead's core 0 alone: every bucket's total becomes its start, a
 * serial walk. Returns the keys placed, which must be the key count. */
static uint32_t starts(void) {
    uint32_t* totals = totals_of();
    for (uint32_t block = 1; block < blocks(); block++) {
        uint32_t first_chunk, end_chunk;
        bs_block(block, blocks(), chunks(), &first_chunk, &end_chunk);
        /* the totals clusters 2 and 3 pushed into my bank: invalidate before reading */
        kmx_invalidate_at(
            REGION_TOTALS,
            skew() + (uint64_t)first_chunk * LINE,
            (uint64_t)(end_chunk - first_chunk) * LINE
        );
    }
    uint32_t running_total = 0;
    for (uint32_t bucket = 0; bucket < buckets(); bucket++) {
        const uint32_t total = totals[bucket];
        totals[bucket] = running_total;
        running_total += total;
    }
    for (uint32_t block = 1; block < blocks(); block++) {
        uint32_t first_chunk, end_chunk;
        bs_block(block, blocks(), chunks(), &first_chunk, &end_chunk);
        const uint64_t offset = skew() + (uint64_t)first_chunk * LINE;
        const uint64_t length = (uint64_t)(end_chunk - first_chunk) * LINE;
        if (length == 0) {
            continue;
        }
        /* Their starts back to them; then flush the lines here, or a dirty line
         * would later be written over the totals they push in the next round. */
        if (kmx_push(
            cluster_of(block),
            REGION_TOTALS,
            offset,
            totals + first_chunk * BS_CHUNK,
            length
        ) != KMX_OK) {
            push_failed = 1;
        }
        kmx_flush_at(REGION_TOTALS, offset, length);
    }
    return running_total;
}

/* The command of step 1 or 3 for the whole Tensix group, field by field (a struct
 * assignment may call memset, which does not link). */
static void build_command(
    km_command_t* command, uint32_t handler, const uint32_t tables[BS_BLOCKS_MAX]
) {
    uint32_t banks = 0;
    for (uint32_t block = 0; block < blocks(); block++) {
        banks |= (
            ((word(BS_WORD_BANKS) >> (8 * (cluster_of(block) - 1))) & 0xFFu)
            << (8 * block)
        );
    }
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = handler;
    command->group = 0u | word(BS_WORD_CORES) << 16; /* cores 0 to cores - 1 */
    command->a0 = word(BS_WORD_KEYS_ADDRESS);
    command->a1 = word(BS_WORD_KEY_COUNT);
    command->a2 = word(BS_WORD_LOG2_BUCKETS) | blocks() << 8;
    command->a3 = banks;
    command->a4 = tables[0];
    command->a5 = tables[1];
    command->a6 = tables[2];
    command->a7 = word(BS_WORD_OUTPUT_ADDRESS);
    command->a8 = command->a9 = 0;
}

/* Submits the command through this cluster's ring to the dispatcher on cluster 0 and
 * waits for its event, both with a timeout: 0 once every core reported it done. */
static uint32_t run_command(const km_command_t* command) {
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int error = kmx_submit(command, &handle);
    while (error == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        error = kmx_submit(command, &handle);
    }
    kmx_event event;
    if (
        error != KMX_OK || kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK
    ) {
        return CODE_RING;
    }
    return event.kind == KM_EVENT_DONE ? 0u
                                       : CODE_COMMAND; /* a timeout or a rejection */
}

int kmx_main(int core) {
    (void)core;
    if (kmx_cluster() != kmx_lead()) {
        /* Clusters 2 and 3: run the lead's kmx_parallel_all calls until its kmx_main
         * ends. */
        const int error = kmx_serve();
        return error != KMX_OK ? CODE_PARALLEL : push_failed ? CODE_PUSH : 0;
    }
    /* From here on: core 0 of the lead, cluster 1. Check the words and regions first.
     */
    const kmx_args* arguments = kmx_args_get();
    const uint32_t x280_cores = (
        arguments->nwords >= BS_WORDS ? word(BS_WORD_X280_CORES) : 0
    );
    const uint32_t expected_clusters = x280_cores == 12
        ? 0xEu
        : 0x2u; /* clusters 1 to 3, or 1 alone */
    if (
        (x280_cores != 1 && x280_cores != 4 && x280_cores != 12) ||
        kmx_clusters() != expected_clusters || word(BS_WORD_LOG2_BUCKETS) < 4 ||
        word(BS_WORD_LOG2_BUCKETS) > 14 || word(BS_WORD_CORES) == 0 ||
        word(BS_WORD_REPEATS) == 0
    ) {
        return CODE_BAD;
    }
    const uint64_t table = (uint64_t)word(BS_WORD_CORES) * row_bytes();
    const uint64_t needed[REGION_COUNT] = {table, table, 2 * row_bytes(), 4 * LINE};
    for (int region = 0; region < REGION_COUNT; region++) {
        uint64_t bytes;
        if (kmx_region(region, &bytes) == 0 || bytes < needed[region]) {
            return CODE_BAD;
        }
    }

    /* Step 0: where each block's tables are. Block 0's are my own regions; with twelve
     * cores, clusters 2 and 3 push theirs to me in a first kmx_parallel_all call. */
    uint32_t counts_offsets[BS_BLOCKS_MAX] = {0, 0, 0};
    uint32_t plan_offsets[BS_BLOCKS_MAX] = {0, 0, 0};
    counts_offsets[0] = (uint32_t)arguments->region[REGION_COUNTS].offset;
    plan_offsets[0] = (uint32_t)arguments->region[REGION_PLAN].offset;
    if (blocks() > 1) {
        if (run_phase(PHASE_HELLO) != KMX_OK || push_failed) {
            return CODE_PARALLEL;
        }
        kmx_invalidate(REGION_PEERS);
        const uint32_t* peers = (const uint32_t*)kmx_region(REGION_PEERS, 0);
        for (uint32_t block = 1; block < blocks(); block++) {
            counts_offsets[block] = peers[16 * cluster_of(block)];
            plan_offsets[block] = peers[16 * cluster_of(block) + 1];
        }
    }

    uint32_t result_words[BS_RESULT_WORDS];
    for (uint32_t i = 0; i < BS_RESULT_WORDS; i++) {
        /* best times start at the most */
        result_words[i] = i < BS_RESULT_PLACED ? 0xFFFFFFFFu : 0;
    }
    km_command_t command;
    for (uint32_t round = 0; round < word(BS_WORD_REPEATS); round++) {
        const uint64_t start = kmx_time();
        /* Step 1, COUNT on the Tensix group: submit the command, wait for its event. */
        kmx_progress(1 + 4 * round);
        build_command(&command, BS_HANDLER_COUNT, counts_offsets);
        uint32_t code = run_command(&command);
        if (code) {
            kmx_result(code, result_words, BS_RESULT_WORDS);
            return 0;
        }
        const uint64_t counted = kmx_time();
        /* Step 2, PLAN on the X280 cores: totals, (to the lead), starts, plan. */
        kmx_progress(2 + 4 * round);
        int error = run_phase(PHASE_TOTALS);
        if (error == KMX_OK && blocks() > 1) {
            error = run_phase(PHASE_SEND);
        }
        if (error != KMX_OK) {
            return CODE_PARALLEL;
        }
        result_words[BS_RESULT_PLACED] = starts();
        if (run_phase(PHASE_PLAN) != KMX_OK) {
            return CODE_PARALLEL;
        }
        if (push_failed) {
            return CODE_PUSH;
        }
        const uint64_t planned = kmx_time();
        /* Step 3, PLACE on the Tensix group, with the plan the cores wrote to DRAM. */
        kmx_progress(3 + 4 * round);
        build_command(&command, BS_HANDLER_PLACE, plan_offsets);
        code = run_command(&command);
        if (code) {
            kmx_result(code, result_words, BS_RESULT_WORDS);
            return 0;
        }
        const uint64_t placed = kmx_time();
        const uint32_t ticks[4] = {
            (uint32_t)(counted - start),
            (uint32_t)(planned - counted),
            (uint32_t)(placed - planned),
            (uint32_t)(placed - start)
        };
        for (uint32_t i = 0; i < 4; i++) {
            result_words[i] = ticks[i] < result_words[i] ? ticks[i] : result_words[i];
        }
        result_words[BS_RESULT_ROUNDS] = round + 1;
    }
    kmx_result(0, result_words, BS_RESULT_WORDS);
    return 0;
}
