/* A map, a heavy map and a two-pass prefix sum on 1, 4, 8 and 12 cores: one image on
 * every work cluster, the lead's kmx_par_all, data pushed across banks (README.md). */
#include <stdint.h>

#include "kmx.h"

/* Regions in the host's order (the same on every cluster); words; result words. */
enum {
    R_IN,
    R_REF_MAP,
    R_REF_HEAVY,
    R_REF_PRE,
    R_MAP,
    R_HEAVY,
    R_PRE,
    R_SUMS,
    R_COUNT
};

enum { W_N, W_REPEATS, W_ROUNDS, W_COUNT };

/* Per pattern (map, heavy map, prefix) and cores (1, 4, 8, 12): times, then words that
 * differ from the 1-core output; then the input push's times. */
enum { T_TIME = 0, T_DIFFER = 12, T_PUSH = 24, N_WORDS = 28 };

#define CHUNK 1024u /* words of one partition: 4 KiB */
#define CHUNKS_MAX 4096u
#define LINE \
    16u /* words: a chunk's sum has a line of its own, so each block's sums push */
#define CODE_BAD 1u
#define CODE_LATE 2u
#define CODE_PUSH 3u

/* What every core runs (the call's word: cores | phase << 8 | region << 16, the
 * region written or sent); a map's f is applied the words' rounds for R_HEAVY. */
enum { P_MAP, P_SUMS, P_PRE, P_SEND };

static const uint32_t X280_CORES[4] = {1, 4, 8, 12};
/* Set by a core whose push failed (globals need .data and an initialiser). */
static volatile uint32_t g_bad __attribute__((section(".data"))) = 0;

static inline uint32_t f(uint32_t x) {
    return (x ^ (x >> 15)) * 2246822519u + 1u;
}

static uint64_t words(void) {
    return kmx_args_get()->words[W_N];
}

static uint64_t chunks(void) {
    return (words() + CHUNK - 1) / CHUNK;
}

/* The clusters in use with `cores` (the lowest first, the lead), and cluster c's
 * index among them; -1 when c takes no part. */
static int index_of(uint32_t x280_cores, int c, uint32_t* count) {
    const uint32_t mask = kmx_clusters();
    *count = x280_cores <= 4 ? 1 : x280_cores / 4;
    uint32_t k = 0;
    for (int i = 1; i < 4; i++) {
        if (mask >> i & 1u) {
            if (i == c) {
                return k < *count ? (int)k : -1;
            }
            k++;
        }
    }
    return -1;
}

/* Cluster k of `count`: chunks [lo, hi), whole groups each. */
static void block(uint32_t k, uint32_t count, uint64_t* lo, uint64_t* hi) {
    *lo = chunks() * k / count;
    *hi = chunks() * (k + 1) / count;
    *hi = *hi < chunks() ? *hi : chunks();
    *lo = *lo < *hi ? *lo : *hi;
}

/* Bytes [at, at + n) of 32-bit words [lo, hi), whole lines (regions are padded). */
static void bytes_of(uint64_t lo, uint64_t hi, uint64_t* at, uint64_t* n) {
    *at = lo * 4;
    *n = ((hi - lo) * 4 + 63) / 64 * 64;
}

static uint64_t end_of(uint64_t c) {
    return (c + 1) * CHUNK < words() ? (c + 1) * CHUNK : words();
}

/* One core's partition of a phase: chunks lo + core, lo + core + parts, ... */
static void step(int cluster, int core, uint64_t arg) {
    const uint32_t x280_cores = arg & 0xFFu, phase = (arg >> 8) & 0xFFu;
    uint32_t count;
    const int k = index_of(x280_cores, cluster, &count);
    const uint32_t parts = x280_cores == 1 ? 1 : 4;
    if (k < 0 || (uint32_t)core >= parts) {
        return;
    }
    uint64_t lo, hi;
    block((uint32_t)k, count, &lo, &hi);
    const uint32_t* in = (const uint32_t*)kmx_region(R_IN, 0);
    uint32_t* sums = (uint32_t*)kmx_region(R_SUMS, 0);
    if (phase == P_SEND) {
        /* not the lead: its quarter of the block's range of a region, to the lead */
        const int r = (int)((arg >> 16) & 0xFFu);
        const uint64_t unit = r == R_SUMS ? LINE : CHUNK;
        uint64_t at, n;
        bytes_of(lo * unit, hi * unit, &at, &n);
        const uint64_t lines = n / 64, per = (lines + 3) / 4, first = per * core;
        if (k == 0 || first >= lines) {
            return;
        }
        at += first * 64;
        n = (lines - first < per ? lines - first : per) * 64;
        const uint8_t* src = (const uint8_t*)kmx_region(r, 0) + at;
        if (kmx_push(kmx_lead(), r, at, src, n) != KMX_OK) {
            g_bad = 1;
        }
        if (r == R_SUMS) {
            kmx_flush_at(r, at, n); /* the lead pushes the bases back over them */
        }
        return;
    }
    if (k != 0 && phase == P_PRE) {
        uint64_t at, n;
        bytes_of(lo * LINE, hi * LINE, &at, &n);
        kmx_invalidate_at(R_SUMS, at, n); /* the bases the lead pushed */
    }
    const int r = (int)((arg >> 16) & 0xFFu);
    const uint32_t rounds = r == R_HEAVY ? kmx_args_get()->words[W_ROUNDS] : 1;
    uint32_t* out = (uint32_t*)kmx_region(r, 0);
    for (uint64_t c = lo + (uint64_t)core; c < hi; c += parts) {
        const uint64_t a = c * CHUNK, b = end_of(c);
        if (k != 0 && phase != P_PRE) {
            kmx_invalidate_at(R_IN, a * 4, (b - a) * 4); /* what the lead pushed */
        }
        if (phase == P_MAP) {
            for (uint64_t i = a; i < b; i++) {
                uint32_t x = in[i];
                for (uint32_t j = 0; j < rounds; j++) {
                    x = f(x);
                }
                out[i] = x;
            }
        } else if (phase == P_SUMS) {
            uint32_t s = 0;
            for (uint64_t i = a; i < b; i++) {
                s += in[i];
            }
            sums[c * LINE] = s;
        } else {
            uint32_t run = sums[c * LINE];
            for (uint64_t i = a; i < b; i++) {
                out[i] = run;
                run += in[i];
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
static int phase(uint32_t x280_cores, uint32_t p, int region) {
    const uint64_t arg = x280_cores | p << 8 | (uint64_t)region << 16;
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

/* The lead's pushes of region r's blocks (in `unit` words per chunk) to the others in
 * use, or its takes of what they sent back. */
static int blocks(uint32_t x280_cores, int r, uint64_t unit, int take) {
    uint32_t count;
    (void)index_of(x280_cores, kmx_lead(), &count);
    const uint32_t mask = kmx_clusters();
    uint32_t k = 0;
    for (int c = 1; c < 4; c++) {
        if (!(mask >> c & 1u)) {
            continue;
        }
        uint64_t lo, hi, at, n;
        block(k, count, &lo, &hi);
        bytes_of(lo * unit, hi * unit, &at, &n);
        if (k > 0 && k < count && n != 0) {
            if (take) {
                kmx_invalidate_at(r, at, n);
            } else {
                const uint8_t* src = (const uint8_t*)kmx_region(r, 0) + at;
                if (kmx_push(c, r, at, src, n) != KMX_OK) {
                    return KMX_BAD;
                }
            }
        }
        k++;
    }
    return KMX_OK;
}

/* A map into region r on `cores`: the input out, the map, the outputs back. */
static int map(uint32_t x280_cores, int r, uint32_t* push) {
    const uint64_t t0 = kmx_time();
    int e = blocks(x280_cores, R_IN, CHUNK, 0);
    *push = (uint32_t)(kmx_time() - t0);
    e = e ? e : phase(x280_cores, P_MAP, r);
    e = e ? e : (x280_cores > 4 ? phase(x280_cores, P_SEND, r) : KMX_OK);
    return e ? e : blocks(x280_cores, r, CHUNK, 1);
}

/* The two-pass prefix sum: the chunk sums to the lead, the bases back out. */
static int prefix(uint32_t x280_cores) {
    const int all = x280_cores > 4;
    int e = blocks(x280_cores, R_IN, CHUNK, 0);
    e = e ? e : phase(x280_cores, P_SUMS, 0);
    e = e ? e : (all ? phase(x280_cores, P_SEND, R_SUMS) : KMX_OK);
    e = e ? e : blocks(x280_cores, R_SUMS, LINE, 1);
    if (e) {
        return e;
    }
    uint32_t* sums = (uint32_t*)kmx_region(R_SUMS, 0);
    uint32_t base = 0;
    for (uint64_t c = 0; c < chunks(); c++) {
        const uint32_t s = sums[c * LINE];
        sums[c * LINE] = base;
        base += s;
    }
    e = blocks(x280_cores, R_SUMS, LINE, 0);
    e = e ? e : phase(x280_cores, P_PRE, R_PRE);
    e = e ? e : (all ? phase(x280_cores, P_SEND, R_PRE) : KMX_OK);
    return e ? e : blocks(x280_cores, R_PRE, CHUNK, 1);
}

static uint32_t differ(const uint32_t* a, const uint32_t* b) {
    uint32_t n = 0;
    for (uint64_t i = 0; i < words(); i++) {
        n += a[i] != b[i];
    }
    return n;
}

static void copy(uint32_t* to, const uint32_t* from) {
    for (uint64_t i = 0; i < words(); i++) {
        to[i] = from[i];
    }
}

int kmx_main(int core) {
    (void)core;
    if (kmx_cluster() != kmx_lead()) {
        const int e = kmx_serve();
        return e != KMX_OK ? (int)CODE_LATE : g_bad ? (int)CODE_PUSH : 0;
    }
    const kmx_args* a = kmx_args_get();
    uint32_t w[N_WORDS]; /* every word is set below */
    if (
        a->nwords < W_COUNT || a->nregions != R_COUNT || a->words[W_REPEATS] == 0 ||
        words() == 0 || chunks() > CHUNKS_MAX
    ) {
        return CODE_BAD;
    }
    for (int i = 0; i < R_COUNT; i++) {
        uint64_t bytes;
        const uint64_t need = (
            i == R_SUMS ? CHUNKS_MAX * 64 : (words() * 4 + 63) / 64 * 64
        );
        if (kmx_region(i, &bytes) == 0 || bytes < need) {
            return CODE_BAD;
        }
    }
    uint32_t* r[R_COUNT];
    for (int i = 0; i < R_COUNT; i++) {
        r[i] = (uint32_t*)kmx_region(i, 0);
    }
    static const int OUT[3] = {R_MAP, R_HEAVY, R_PRE};
    for (int m = 0; m < 4; m++) {
        for (int p = 0; p < 3; p++) {
            uint32_t best = 0xFFFFFFFFu;
            for (uint32_t rep = 0; rep < a->words[W_REPEATS]; rep++) {
                kmx_progress(1 + 3 * m + p);
                kmx_invalidate(R_IN); /* every run from a cold cache */
                kmx_invalidate(OUT[p]);
                kmx_flush(R_SUMS); /* no dirty line where the others' sums land */
                const uint64_t t0 = kmx_time();
                const int e = (
                    p < 2
                        ? map(X280_CORES[m], OUT[p], &w[T_PUSH + m])
                        : prefix(X280_CORES[m])
                );
                const uint32_t t = (uint32_t)(kmx_time() - t0);
                if (e != KMX_OK) {
                    return e == KMX_BAD ? (int)CODE_PUSH : (int)CODE_LATE;
                }
                best = t < best ? t : best;
            }
            w[T_TIME + 4 * p + m] = best;
            if (m == 0) {
                copy(r[OUT[p] - 3], r[OUT[p]]); /* the 1-core output: the reference */
            }
            w[T_DIFFER + 4 * p + m] = differ(r[OUT[p]], r[OUT[p] - 3]);
        }
    }
    kmx_result(g_bad ? CODE_PUSH : 0, w, N_WORDS);
    return 0;
}
