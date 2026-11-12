/* Prefix sums of region 0 (counts, cores x bins; words: cores, bins) into region 1 on
 * four harts. Code 0 and the sum of counts; 1 bad arguments; 2 kmx_par failed. */
#include <stdint.h>

#include "kmx.h"

#define BINS_MAX 4096u

typedef struct {
    const volatile uint32_t* counts;
    volatile uint32_t* offsets;
    uint32_t cores, bins;
    uint32_t total[4]; /* hart h: the counts in its bins */
} job_t;

/* Pass 1: hart h's bin totals over all cores, kept as exclusive prefixes in start[]. */
static uint32_t start[4][BINS_MAX / 4] __attribute__((section(".data"))) = {{1}};

static void totals(int hart, void* arg) {
    job_t* j = (job_t*)arg;
    const uint32_t chunk = j->bins / 4, b0 = (uint32_t)hart * chunk;
    uint32_t* s = start[hart];
    for (uint32_t b = 0; b < chunk; b++) {
        s[b] = 0;
    }
    for (uint32_t c = 0; c < j->cores; c++) {
        for (uint32_t b = 0; b < chunk; b++) {
            s[b] += j->counts[c * j->bins + b0 + b];
        }
    }
    uint32_t run = 0;
    for (uint32_t b = 0; b < chunk; b++) {
        const uint32_t t = s[b];
        s[b] = run;
        run += t;
    }
    j->total[hart] = run;
}

/* Pass 2: the offsets of hart h's bins, after the totals of the harts before it. */
static void offsets(int hart, void* arg) {
    job_t* j = (job_t*)arg;
    const uint32_t chunk = j->bins / 4, b0 = (uint32_t)hart * chunk;
    uint32_t* s = start[hart];
    uint32_t base = 0;
    for (int h = 0; h < hart; h++) {
        base += j->total[h];
    }
    for (uint32_t c = 0; c < j->cores; c++) {
        for (uint32_t b = 0; b < chunk; b++) {
            j->offsets[c * j->bins + b0 + b] = base + s[b];
            s[b] += j->counts[c * j->bins + b0 + b];
        }
    }
}

int kmx_main(int hart) {
    (void)hart;
    const kmx_args* a = kmx_args_get();
    uint64_t in_bytes, out_bytes;
    job_t j;
    j.counts = (const volatile uint32_t*)kmx_region(0, &in_bytes);
    j.offsets = (volatile uint32_t*)kmx_region(1, &out_bytes);
    j.cores = a->nwords >= 2 ? a->words[0] : 0;
    j.bins = a->nwords >= 2 ? a->words[1] : 0;
    const uint64_t need = (uint64_t)j.cores * j.bins * 4;
    if (!j.counts || !j.offsets || j.bins == 0 || j.bins % 64 != 0 ||
        j.bins > BINS_MAX || need > in_bytes || need > out_bytes) {
        return 1; /* the code the host sees: bad arguments */
    }
    kmx_progress(1);
    kmx_take(0);
    kmx_progress(2);
    int e = kmx_par(totals, &j);
    kmx_progress(3);
    if (e == KMX_OK) {
        e = kmx_par(offsets, &j);
    }
    kmx_progress(4);
    kmx_give(1);
    kmx_progress(5);
    const uint32_t sum = j.total[0] + j.total[1] + j.total[2] + j.total[3];
    kmx_result(e == KMX_OK ? 0u : 2u, &sum, 1);
    return 0;
}
