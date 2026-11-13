/* Bucket sort, X280 side: the THREAD, one image on each work cluster taking part. Core
 * 0 of cluster 1 drives the whole program; the plan runs on 1, 4 or 12 cores. */
#include <stdint.h>

#include "bucket_sort.h"
#include "kmx.h"

/* The regions, in the host's order. Each cluster has its own copy, in its own bank. */
enum { R_COUNTS, R_PLAN, R_TOTALS, R_PEERS, R_COUNT };

/* What the planning cores do in one call (step() below). */
enum { P_HELLO, P_TOTALS, P_SEND, P_PLAN };

/* The thread's codes (0: done). */
enum { CODE_BAD = 1, CODE_PAR, CODE_PUSH, CODE_RING, CODE_COMMAND };

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command: longer than its deadline */
#define LINE 64u                        /* bytes: one cache line, one chunk of a row */

/* Set by a core whose push failed (globals need .data and an initialiser). */
static volatile uint32_t g_bad __attribute__((section(".data"))) = 0;

static uint32_t word(int i) {
    return kmx_args_get()->words[i];
}

static uint32_t buckets(void) {
    return 1u << word(BS_W_LOG2_B);
}

static uint32_t chunks(void) {
    return buckets() / BS_CHUNK;
}

static uint32_t blocks(void) {
    return word(BS_W_X280_CORES) == 12 ? 3u : 1u;
}

/* Bytes from one table row to the next: B counters of 4 bytes. */
static uint64_t row_bytes(void) {
    return (uint64_t)buckets() * 4;
}

/* Where the totals start in their region: half a row away, in the cluster's cache of
 * one 128 KiB way, from every row of the counts and plan beside them (README.md). */
static uint64_t skew(void) {
    const uint64_t row = row_bytes();
    const uintptr_t gap = (
        (uintptr_t)kmx_region(R_TOTALS, 0) - (uintptr_t)kmx_region(R_COUNTS, 0)
    );
    return (row / 2 + row - gap % row) % row / LINE * LINE;
}

static uint32_t* totals_of(void) {
    return (uint32_t*)((uint8_t*)kmx_region(R_TOTALS, 0) + skew());
}

/* Block k is planned by the k-th cluster running this image (cluster 1, the lead,
 * plans block 0); these two functions turn one into the other. */
static uint32_t block_of(int cluster) {
    uint32_t k = 0;
    for (int c = 1; c < cluster; c++) {
        k += (kmx_clusters() >> c) & 1u;
    }
    return k;
}

static int cluster_of(uint32_t block) {
    for (int c = 1; c < 4; c++) {
        if (((kmx_clusters() >> c) & 1u) && block_of(c) == block) {
            return c;
        }
    }
    return 0;
}

/* Step 2, on every planning core: `arg` is the phase | cores per cluster << 8. Each
 * cluster plans its block of buckets, each of its cores a quarter of that block. */
static void step(int cluster, int core, uint64_t arg) {
    const uint32_t phase = arg & 0xFFu, parts = (arg >> 8) & 0xFFu;
    if ((uint32_t)core >= parts) {
        return;
    }
    const uint32_t k = block_of(cluster), b = buckets(), rows = word(BS_W_CORES);
    const uint64_t row = row_bytes();
    uint32_t lo, hi;
    bs_block(k, blocks(), chunks(), &lo, &hi);
    /* This cluster's own copy of each region, in its own bank, read through its cache.
     */
    const uint32_t* counts = (const uint32_t*)kmx_region(R_COUNTS, 0);
    uint32_t* plan = (uint32_t*)kmx_region(R_PLAN, 0);
    uint32_t* totals = totals_of();
    if (phase == P_HELLO || phase == P_SEND) {
        /* One core per cluster other than the lead pushes a line to the lead. */
        if (k == 0 || core != 0) {
            return;
        }
        if (phase == P_HELLO) {
            /* Step 0: the lead cannot see my regions' DRAM addresses; I push them
             * into my line of its `peers` region, for the commands it builds. */
            uint32_t* line = (uint32_t*)kmx_region(R_PEERS, 0) + 16 * cluster;
            line[0] = (uint32_t)kmx_args_get()->region[R_COUNTS].offset;
            line[1] = (uint32_t)kmx_args_get()->region[R_PLAN].offset;
            if (kmx_push(kmx_lead(), R_PEERS, LINE * cluster, line, LINE) != KMX_OK) {
                g_bad = 1;
            }
            return;
        }
        /* Step 2b: my block's bucket totals to the lead's bank (it cannot read mine),
         * then flush them: the lead pushes the starts over these very lines. */
        if (hi > lo) {
            const uint64_t at = skew() + (uint64_t)lo * LINE;
            const uint64_t n = (uint64_t)(hi - lo) * LINE;
            if (
                kmx_push(kmx_lead(), R_TOTALS, at, totals + lo * BS_CHUNK, n) != KMX_OK
            ) {
                g_bad = 1;
            }
            kmx_flush_at(R_TOTALS, at, n);
        }
        return;
    }
    /* My part of the block: a run of whole chunks, so no two cores share a line. */
    uint32_t a, z;
    bs_block((uint32_t)core, parts, hi - lo, &a, &z);
    const uint32_t first = (lo + a) * BS_CHUNK, end = (lo + z) * BS_CHUNK;
    const uint64_t at = (uint64_t)first * 4, n = (uint64_t)(end - first) * 4;
    if (n == 0) {
        return;
    }
    if (phase == P_TOTALS) {
        /* Step 2a: my buckets' totals, row after row (a row is read as one stream).
         * The cores wrote the rows over the NoC, past this cache: invalidate first. */
        for (uint32_t x = first; x < end; x++) {
            totals[x] = 0;
        }
        for (uint32_t r = 0; r < rows; r++) {
            kmx_invalidate_at(R_COUNTS, r * row + at, n);
            const uint32_t* in = counts + (uint64_t)r * b;
            for (uint32_t x = first; x < end; x++) {
                totals[x] += in[x];
            }
        }
        return;
    }
    /* Step 2d: from each bucket's start (from the lead), row after row: core r's first
     * key of a bucket goes after those of cores 0 to r - 1 (totals become the ends). */
    if (k != 0) {
        kmx_invalidate_at(R_TOTALS, skew() + at, n); /* the starts the lead pushed */
    }
    for (uint32_t r = 0; r < rows; r++) {
        const uint32_t* in = counts + (uint64_t)r * b;
        uint32_t* out = plan + (uint64_t)r * b;
        for (uint32_t x = first; x < end; x++) {
            out[x] = totals[x];
            totals[x] += in[x];
        }
        /* The cores read the plan over the NoC, from DRAM: write the lines back. */
        kmx_flush_at(R_PLAN, r * row + at, n);
    }
}

typedef struct {
    uint64_t arg;
} local_t;

static void local_step(int core, void* arg) {
    step(kmx_lead(), core, ((const local_t*)arg)->arg);
}

/* One phase of step 2 on the cores the host asked for: core 0 alone, cluster 1's four
 * (kmx_par), or all twelve of clusters 1 to 3 (kmx_par_all). Each call is timed. */
static int phase(uint32_t p) {
    const uint32_t x280_cores = word(BS_W_X280_CORES);
    const uint64_t arg = p | (uint64_t)(x280_cores == 1 ? 1u : 4u) << 8;
    if (x280_cores == 1) {
        step(kmx_lead(), 0, arg);
        return KMX_OK;
    }
    if (x280_cores == 4) {
        local_t l = {arg};
        return kmx_par(local_step, &l);
    }
    return kmx_par_all(step, arg);
}

/* Step 2c, on the lead's core 0 alone: every bucket's total becomes its start, a
 * serial walk over B words. Returns the keys placed, which must be N. */
static uint32_t starts(void) {
    uint32_t* totals = totals_of();
    for (uint32_t k = 1; k < blocks(); k++) {
        uint32_t lo, hi;
        bs_block(k, blocks(), chunks(), &lo, &hi);
        /* the totals clusters 2 and 3 pushed into my bank: invalidate before reading */
        kmx_invalidate_at(
            R_TOTALS, skew() + (uint64_t)lo * LINE, (uint64_t)(hi - lo) * LINE
        );
    }
    uint32_t run = 0;
    for (uint32_t b = 0; b < buckets(); b++) {
        const uint32_t t = totals[b];
        totals[b] = run;
        run += t;
    }
    for (uint32_t k = 1; k < blocks(); k++) {
        uint32_t lo, hi;
        bs_block(k, blocks(), chunks(), &lo, &hi);
        const uint64_t at = skew() + (uint64_t)lo * LINE;
        const uint64_t n = (uint64_t)(hi - lo) * LINE;
        if (n == 0) {
            continue;
        }
        /* Their starts back to them; then flush the lines here, or a dirty line
         * would later be written over the totals they push in the next round. */
        if (
            kmx_push(cluster_of(k), R_TOTALS, at, totals + lo * BS_CHUNK, n) != KMX_OK
        ) {
            g_bad = 1;
        }
        kmx_flush_at(R_TOTALS, at, n);
    }
    return run;
}

/* The command of step 1 or 3 for the whole Tensix group, field by field (a struct
 * assignment may call memset, which does not link). */
static void command(
    km_command_t* r, uint32_t handler, const uint32_t table[BS_BLOCKS_MAX]
) {
    uint32_t banks = 0;
    for (uint32_t k = 0; k < blocks(); k++) {
        banks |= ((word(BS_W_BANKS) >> (8 * (cluster_of(k) - 1))) & 0xFFu) << (8 * k);
    }
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = handler;
    r->group = 0u | word(BS_W_CORES) << 16; /* cores 0 to cores - 1 */
    r->a0 = word(BS_W_KEYS);
    r->a1 = word(BS_W_KEYS_N);
    r->a2 = word(BS_W_LOG2_B) | blocks() << 8;
    r->a3 = banks;
    r->a4 = table[0];
    r->a5 = table[1];
    r->a6 = table[2];
    r->a7 = word(BS_W_OUT);
    r->a8 = r->a9 = 0;
}

/* Submits r through this cluster's command ring to the dispatcher on cluster 0,
 * and waits for its event, both with a timeout: 0 once every core reported it done. */
static uint32_t run_command(const km_command_t* r) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int e = kmx_submit(r, &handle);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_submit(r, &handle);
    }
    kmx_event ev;
    if (e != KMX_OK || kmx_wait(handle, kmx_time() + WAIT_TICKS, &ev) != KMX_OK) {
        return CODE_RING;
    }
    return ev.kind == KM_EV_DONE ? 0u : CODE_COMMAND; /* a timeout or a rejection */
}

int kmx_main(int core) {
    (void)core;
    if (kmx_cluster() != kmx_lead()) {
        /* Clusters 2 and 3: run the lead's kmx_par_all calls until its kmx_main ends.
         */
        const int e = kmx_serve();
        return e != KMX_OK ? CODE_PAR : g_bad ? CODE_PUSH : 0;
    }
    /* From here on: core 0 of the lead, cluster 1. Check the words and regions first.
     */
    const kmx_args* a = kmx_args_get();
    const uint32_t x280_cores = a->nwords >= BS_WORDS ? word(BS_W_X280_CORES) : 0;
    const uint32_t want = x280_cores == 12 ? 0xEu
                                           : 0x2u; /* clusters 1 to 3, or 1 alone */
    if (
        (x280_cores != 1 && x280_cores != 4 && x280_cores != 12) ||
        kmx_clusters() != want || word(BS_W_LOG2_B) < 4 || word(BS_W_LOG2_B) > 14 ||
        word(BS_W_CORES) == 0 || word(BS_W_REPEATS) == 0
    ) {
        return CODE_BAD;
    }
    const uint64_t table = (uint64_t)word(BS_W_CORES) * row_bytes();
    const uint64_t need[R_COUNT] = {table, table, 2 * row_bytes(), 4 * LINE};
    for (int i = 0; i < R_COUNT; i++) {
        uint64_t bytes;
        if (kmx_region(i, &bytes) == 0 || bytes < need[i]) {
            return CODE_BAD;
        }
    }

    /* Step 0: where each block's tables are. Block 0's are my own regions; with twelve
     * cores, clusters 2 and 3 push theirs to me in a first kmx_par_all call. */
    uint32_t counts_at[BS_BLOCKS_MAX] = {0, 0, 0}, plan_at[BS_BLOCKS_MAX] = {0, 0, 0};
    counts_at[0] = (uint32_t)a->region[R_COUNTS].offset;
    plan_at[0] = (uint32_t)a->region[R_PLAN].offset;
    if (blocks() > 1) {
        if (phase(P_HELLO) != KMX_OK || g_bad) {
            return CODE_PAR;
        }
        kmx_invalidate(R_PEERS);
        const uint32_t* peers = (const uint32_t*)kmx_region(R_PEERS, 0);
        for (uint32_t k = 1; k < blocks(); k++) {
            counts_at[k] = peers[16 * cluster_of(k)];
            plan_at[k] = peers[16 * cluster_of(k) + 1];
        }
    }

    uint32_t out[BS_OUT_WORDS];
    for (uint32_t i = 0; i < BS_OUT_WORDS; i++) {
        out[i] = i < BS_O_PLACED ? 0xFFFFFFFFu : 0; /* best times start at the most */
    }
    km_command_t r;
    for (uint32_t round = 0; round < word(BS_W_REPEATS); round++) {
        const uint64_t t0 = kmx_time();
        /* Step 1, COUNT on the Tensix group: submit the command, wait for its event. */
        kmx_progress(1 + 4 * round);
        command(&r, BS_HANDLER_COUNT, counts_at);
        uint32_t code = run_command(&r);
        if (code) {
            kmx_result(code, out, BS_OUT_WORDS);
            return 0;
        }
        const uint64_t t1 = kmx_time();
        /* Step 2, PLAN on the X280 cores: totals, (to the lead), starts, plan. */
        kmx_progress(2 + 4 * round);
        int e = phase(P_TOTALS);
        if (e == KMX_OK && blocks() > 1) {
            e = phase(P_SEND);
        }
        if (e != KMX_OK) {
            return CODE_PAR;
        }
        out[BS_O_PLACED] = starts();
        if (phase(P_PLAN) != KMX_OK) {
            return CODE_PAR;
        }
        if (g_bad) {
            return CODE_PUSH;
        }
        const uint64_t t2 = kmx_time();
        /* Step 3, PLACE on the Tensix group, with the plan the cores wrote to DRAM. */
        kmx_progress(3 + 4 * round);
        command(&r, BS_HANDLER_PLACE, plan_at);
        code = run_command(&r);
        if (code) {
            kmx_result(code, out, BS_OUT_WORDS);
            return 0;
        }
        const uint64_t t3 = kmx_time();
        const uint32_t took[4] = {
            (uint32_t)(t1 - t0),
            (uint32_t)(t2 - t1),
            (uint32_t)(t3 - t2),
            (uint32_t)(t3 - t0)
        };
        for (uint32_t i = 0; i < 4; i++) {
            out[i] = took[i] < out[i] ? took[i] : out[i];
        }
        out[BS_O_ROUNDS] = round + 1;
    }
    kmx_result(0, out, BS_OUT_WORDS);
    return 0;
}
