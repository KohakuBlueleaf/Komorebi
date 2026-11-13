/* Three patterns on the four cores with kmx_parallel, each run with one partition and
 * with four, timed: a map, a reduction, a two-pass prefix sum. See README.md. */
#include <stdint.h>

#include "kmx.h"
#include "vmap.h"

/* Regions in the host's order; words; result words (timings, then 64-bit sums). */
enum {
    R_IN,
    R_MAP1,
    R_MAP4,
    R_MAPQ,
    R_ALIAS,
    R_VMAP1,
    R_VMAP4,
    R_HEAVY1,
    R_HEAVY4,
    R_SEQUENCE,
    R_PRE1,
    R_PRE4,
    R_COUNTERS,
    R_COUNT
};

enum { W_N, W_REPEATS, W_ROUNDS, W_COUNT };

enum {
    T_MAP1,
    T_MAP4,
    T_MAPQ,
    T_ALIAS,
    T_VMAP1,
    T_VMAP4,
    T_HEAVY1,
    T_HEAVY4,
    T_SUM1,
    T_SUM4,
    T_PACKED,
    T_PADDED,
    T_SEQUENCE,
    T_PRE1,
    T_PRE4,
    T_INVALIDATE1,
    T_INVALIDATE4,
    T_FLUSH1,
    T_FLUSH4,
    T_PARALLEL,
    S_SUM1,
    N_WORDS = S_SUM1 + 8
};

#define CHUNK 1024u /* words of one partition: 4 KiB, whole 64-byte lines */
#define CHUNKS_MAX 4096u
#define PARALLEL_CALLS 1000u
/* The cluster's L3 is one 128 KiB way: the aliased map writes this far into its
 * region, which then starts a whole number of ways from the input. */
#define HALF_WAY_WORDS ((64u << 10) / 4u)

/* Per-core results, one cache line each (globals need .data and an initialiser). */
static uint64_t g_part[4][8] __attribute__((section(".data"), aligned(64))) = {{1}};
/* The prefix sum's per-chunk sums, then each chunk's base. */
static uint32_t g_chunk[CHUNKS_MAX] __attribute__((section(".data"))) = {1};

typedef struct {
    const uint32_t* in;
    uint32_t* out;
    uint64_t n;
    uint32_t parts;         /* 1 or 4: core h takes chunks h, h + parts, ... */
    uint32_t rounds;        /* f() applied this many times per word */
    uint32_t quarters;      /* 1: core h takes the h-th quarter instead */
    uint32_t stride;        /* words of uint64_t between two cores' counters */
    volatile uint64_t* ctr; /* the reduction's counters in a region, or 0 */
} job_t;

static inline uint64_t chunks(const job_t* j) {
    return (j->n + CHUNK - 1) / CHUNK;
}

/* Core h's first chunk; its next ones are `parts` apart. None past `parts`. */
static inline uint64_t first(const job_t* j, int core) {
    return (uint32_t)core < j->parts ? (uint64_t)core : chunks(j);
}

static inline uint64_t end(const job_t* j, uint64_t c) {
    return (c + 1) * CHUNK < j->n ? (c + 1) * CHUNK : j->n;
}

static inline uint32_t f(uint32_t x) {
    return (x ^ (x >> 15)) * 2246822519u + 1u;
}

/* (a) map: out[i] = f(in[i]), f applied `rounds` times. The job's fields are read
 * once: a uint32_t store may alias j->rounds, which would be re-read every word. */
static void map_range(const job_t* j, uint64_t lo, uint64_t hi) {
    const uint32_t* in = j->in;
    uint32_t* out = j->out;
    const uint32_t rounds = j->rounds;
    for (uint64_t i = lo; i < hi; i++) {
        uint32_t x = in[i];
        for (uint32_t r = 0; r < rounds; r++) {
            x = f(x);
        }
        out[i] = x;
    }
}

static void map_part(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    if (j->quarters) {
        const uint64_t h = (uint64_t)core;
        map_range(j, j->n * h / 4, j->n * (h + 1) / 4);
        return;
    }
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        map_range(j, c * CHUNK, end(j, c));
    }
}

/* The same map (one round) with the vector unit, vmap.c. */
static void vmap_part(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        vmap(j->out + c * CHUNK, j->in + c * CHUNK, end(j, c) - c * CHUNK);
    }
}

/* (b) reduction: the core's sum in a register, written once to its line; or, with
 * `ctr`, added word by word to a counter in a region. */
static void sum_part(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    uint64_t s = 0;
    volatile uint64_t* ctr = j->ctr ? j->ctr + (uint64_t)core * j->stride : 0;
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        for (uint64_t i = c * CHUNK; i < end(j, c); i++) {
            if (ctr) {
                *ctr += j->in[i];
            } else {
                s += j->in[i];
            }
        }
    }
    if ((uint32_t)core < j->parts) {
        g_part[core][0] = s;
    }
}

/* (c) prefix, pass 1: each chunk's sum. */
static void chunk_sums(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    /* my chunks of CHUNK words: core, core + parts, ... (parts: the cores in use) */
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        uint32_t s = 0;
        for (uint64_t i = c * CHUNK; i < end(j, c); i++) {
            s += j->in[i];
        }
        g_chunk[c] = s;
    }
}

/* Pass 2: out[i] = the chunk's base + the chunk's words before i. */
static void chunk_prefix(int core, void* arg) {
    const job_t* j = (const job_t*)arg;
    for (uint64_t c = first(j, core); c < chunks(j); c += j->parts) {
        uint32_t run = g_chunk[c];
        for (uint64_t i = c * CHUNK; i < end(j, c); i++) {
            j->out[i] = run;
            run += j->in[i];
        }
    }
}

static void nothing(int core, void* arg) {
    (void)core;
    (void)arg;
}

/* Runs fn on core 0 alone or on all four; KMX_OK, or kmx_parallel's error. */
static int on(void (*fn)(int, void*), job_t* j) {
    if (j->parts == 1) {
        fn(0, j);
        return KMX_OK;
    }
    return kmx_parallel(fn, j);
}

/* The two-pass exclusive prefix sum (mod 2^32); between the passes core 0 turns the
 * chunk sums into bases. */
static int prefix(job_t* j) {
    const int e = on(chunk_sums, j);
    uint32_t base = 0;
    for (uint64_t c = 0; c < chunks(j); c++) {
        const uint32_t s = g_chunk[c];
        g_chunk[c] = base;
        base += s;
    }
    return e != KMX_OK ? e : on(chunk_prefix, j);
}

/* The one-pass prefix sum on core 0: the baseline two passes are measured against. */
static void prefix_sequence(job_t* j) {
    uint32_t run = 0;
    for (uint64_t i = 0; i < j->n; i++) {
        j->out[i] = run;
        run += j->in[i];
    }
}

/* Region g_sweep swept by core 0 alone (the others idle): one core's flush or
 * invalidate. */
static int g_sweep __attribute__((section(".data"))) = -1;

static void sweep_alone(int core, void* arg) {
    (void)arg;
    if (core == 0) {
        kmx_invalidate(g_sweep); /* inside kmx_parallel: this core alone */
    }
}

typedef enum { MAP, VMAP, SUM, PREFIX, SEQUENCE } kind_t;

/* Best of `repeats` runs of one pattern, each from a cold cache (input and output
 * swept first); the output is flushed after. KMX_OK, or kmx_parallel's error. */
static int timed(kind_t k, job_t* j, int out, uint32_t repeats, uint32_t* best) {
    *best = 0xFFFFFFFFu;
    for (uint32_t r = 0; r < repeats; r++) {
        kmx_invalidate(R_IN);
        if (out >= 0) {
            kmx_invalidate(out);
        }
        const uint64_t t0 = kmx_time();
        int e = KMX_OK;
        if (k == MAP || k == VMAP) {
            e = on(k == MAP ? map_part : vmap_part, j);
        } else if (k == SUM) {
            e = on(sum_part, j);
        } else if (k == PREFIX) {
            e = prefix(j);
        } else {
            prefix_sequence(j);
        }
        const uint32_t t = (uint32_t)(kmx_time() - t0);
        if (e != KMX_OK) {
            return e;
        }
        *best = t < *best ? t : *best;
    }
    if (out >= 0) {
        kmx_flush(out);
    }
    return KMX_OK;
}

static void put64(uint32_t* w, uint64_t v) {
    w[0] = (uint32_t)v;
    w[1] = (uint32_t)(v >> 32);
}

/* The maps: output region, result word, cores, heavy (f `rounds` times), quarters,
 * the output's offset in words. */
static const uint32_t MAPS[8][6] = {
    {R_MAP1, T_MAP1, 1, 0, 0, 0},
    {R_MAP4, T_MAP4, 4, 0, 0, 0},
    {R_MAPQ, T_MAPQ, 4, 0, 1, 0},
    {R_ALIAS, T_ALIAS, 1, 0, 0, HALF_WAY_WORDS},
    {R_VMAP1, T_VMAP1, 1, 0, 0, 0},
    {R_VMAP4, T_VMAP4, 4, 0, 0, 0},
    {R_HEAVY1, T_HEAVY1, 1, 1, 0, 0},
    {R_HEAVY4, T_HEAVY4, 4, 1, 0, 0},
};

int kmx_main(int core) {
    (void)core;
    const kmx_args* a = kmx_args_get();
    uint32_t w[N_WORDS]; /* every word is set below (a zeroing loop may call memset) */
    if (
        a->nwords < W_COUNT || a->nregions != R_COUNT || a->words[W_REPEATS] == 0 ||
        a->words[W_N] == 0 || a->words[W_N] > CHUNKS_MAX * CHUNK
    ) {
        return 1;
    }
    const uint64_t n = a->words[W_N];
    const uint32_t repeats = a->words[W_REPEATS];
    void* r[R_COUNT];
    for (int i = 0; i < R_COUNT; i++) {
        uint64_t bytes;
        r[i] = kmx_region(i, &bytes);
        uint64_t need = n * 4 + (i == R_ALIAS ? HALF_WAY_WORDS * 4 : 0);
        need = i == R_COUNTERS ? 4 * 64 : need;
        if (r[i] == 0 || bytes < need) {
            return 1;
        }
    }
    job_t j = {(const uint32_t*)r[R_IN], 0, n, 1, 1, 0, 0, 0};
    for (int m = 0; m < 8; m++) {
        kmx_progress(1 + m);
        j.out = (uint32_t*)r[MAPS[m][0]] + MAPS[m][5];
        j.parts = MAPS[m][2];
        j.rounds = MAPS[m][3] ? a->words[W_ROUNDS] : 1;
        j.quarters = MAPS[m][4];
        const kind_t k = m == 4 || m == 5 ? VMAP : MAP;
        if (timed(k, &j, (int)MAPS[m][0], repeats, &w[MAPS[m][1]])) {
            return 2;
        }
    }
    /* reductions: in a register (one core, four), then in counters (packed, padded) */
    j.out = 0;
    j.quarters = 0;
    volatile uint64_t* ctr = (volatile uint64_t*)r[R_COUNTERS];
    for (int m = 0; m < 4; m++) {
        kmx_progress(9 + m);
        j.parts = m == 0 ? 1 : 4;
        j.ctr = m < 2 ? 0 : ctr;
        j.stride = m == 2 ? 1 : 8;
        for (uint32_t i = 0; i < 32; i++) {
            ctr[i] = 0;
        }
        if (timed(SUM, &j, -1, repeats, &w[T_SUM1 + m])) {
            return 2;
        }
        uint64_t s = 0;
        for (uint32_t h = 0; h < j.parts; h++) {
            s += m < 2 ? g_part[h][0] : ctr[h * j.stride];
        }
        put64(&w[S_SUM1 + 2 * m], m < 2 ? s : s / repeats); /* counters kept adding */
    }
    j.ctr = 0;
    static const int pre_out[3] = {R_SEQUENCE, R_PRE1, R_PRE4};
    for (int m = 0; m < 3; m++) {
        kmx_progress(13 + m);
        j.out = (uint32_t*)r[pre_out[m]];
        j.parts = m == 2 ? 4 : 1;
        if (timed(
            m == 0 ? SEQUENCE : PREFIX, &j, pre_out[m], repeats, &w[T_SEQUENCE + m]
        )) {
            return 2;
        }
    }
    /* flush and invalidate: a region just read (clean lines), then one just written
     * (dirty), swept by core 0 alone and by all four */
    kmx_progress(16);
    j.out = (uint32_t*)r[R_MAP4];
    j.parts = 4;
    j.rounds = 1;
    for (int m = 0; m < 2; m++) {
        const int id = m == 0 ? R_IN : R_MAP4;
        for (int x280_cores = 1; x280_cores <= 4; x280_cores += 3) {
            if (on(m == 0 ? sum_part : map_part, &j) != KMX_OK) {
                return 2;
            }
            const uint64_t t0 = kmx_time();
            if (x280_cores == 4) {
                kmx_invalidate(id);
            } else {
                g_sweep = id;
                if (kmx_parallel(sweep_alone, 0) != KMX_OK) {
                    return 2;
                }
            }
            w[(m ? T_FLUSH1 : T_INVALIDATE1) + x280_cores / 4] = (uint32_t)(
                kmx_time() - t0
            );
        }
    }
    kmx_progress(17);
    const uint64_t t0 = kmx_time();
    for (uint32_t i = 0; i < PARALLEL_CALLS; i++) {
        if (kmx_parallel(nothing, 0) != KMX_OK) {
            return 2;
        }
    }
    w[T_PARALLEL] = (uint32_t)(kmx_time() - t0);
    kmx_result(0, w, N_WORDS);
    return 0;
}
