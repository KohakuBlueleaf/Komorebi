/* Rates probe, four harts: read, write and Flush64 rates over a region of the cluster's
 * DRAM bank, then the prefix-sum job over cores x bins counters (result words RT_*). */
#include <stdint.h>

#include "probe.h"

typedef struct {
    uint64_t run_value;
    uint64_t region;      /* bank address of the rate region */
    uint64_t sizes[2];    /* bytes of the two passes, multiples of 4 x 512; 0 = skip */
    uint64_t counts, out; /* bank addresses of the job's input and output */
    uint64_t cores, bins; /* bins a multiple of 4 x 16, at most RT_BINS_MAX */
    uint64_t reps;        /* job repetitions, at most RT_REPS_MAX */
    uint64_t timeout;     /* mtime ticks for the whole run */
} rates_args_t;

#define RT_BINS_MAX 4096u
#define RT_REPS_MAX 8u
#define RT_PHASES 6u

/* Phases of one size pass, in order. */
enum {
    RT_FLUSH_CLEAN = 0, /* Flush64 over lines that are not dirty */
    RT_READ_COLD = 1,   /* vector loads right after that flush */
    RT_READ_AGAIN = 2,  /* the same loads once more */
    RT_WRITE = 3,       /* vector stores of a pattern */
    RT_FLUSH_DIRTY = 4, /* Flush64 over the written lines */
    RT_READ_SCALAR = 5  /* 8-byte loads, after the flush */
};

/* Result words (32-bit). */
#define RT_STATUS 0u
#define RT_TICKS 8u     /* + pass * RT_PHASES + phase: mtime ticks, slowest hart */
#define RT_SUMS 24u     /* + hart: low word of hart's sum of the cold read of pass 0 */
#define RT_JOB 32u      /* + rep: ticks of the job, take to give */
#define RT_JOB_TAKE 40u /* + rep: ticks of the take alone */

/* Barrier words in LIM: hart h writes its arrival (the phase) at word h. */
#define RT_ARRIVE 0x180u
#define RT_RELEASE 0x1C0u /* hart 0: word 0 the released phase, word 2/3 its mtime */
#define RT_TOTALS 0x1D0u  /* the job: hart h's chunk total at word h */

/* Arrives at barrier `n` and waits until hart 0 releases it; hart 0 releases once
 * every hart arrived. Returns 0 or an error; the release time in *t0. */
static uint32_t barrier(pr_t* p, uint32_t n, uint64_t* t0) {
    volatile uint32_t* arrive = pr_word(p, RT_ARRIVE);
    volatile uint32_t* release = pr_word(p, RT_RELEASE);
    arrive[p->hart] = n;
    x280_fence_ow();
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (p->hart == 0) {
            if (arrive[1] >= n && arrive[2] >= n && arrive[3] >= n) {
                const uint64_t t = x280_mtime();
                release[2] = (uint32_t)t;
                release[3] = (uint32_t)(t >> 32);
                x280_fence_ow();
                release[0] = n;
                x280_fence_ow();
                *t0 = t;
                return 0;
            }
        } else if (release[0] >= n) {
            *t0 = release[2] | (uint64_t)release[3] << 32;
            return 0;
        }
        if ((spins & 255u) == 0) {
            const uint32_t e = pr_check(p);
            if (e) {
                return e;
            }
        }
    }
}

/* Sum of the 64-bit words of [p, p + n) with 512-byte vector loads; n >= 512. */
static uint64_t vsum(uintptr_t p, uint64_t n) {
    const uintptr_t end = p + n;
    uint64_t s;
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetvli zero, %[vl], e64, m8, ta, ma\n"
        "vmv.v.i v16, 0\n"
        "1:\n"
        "vle64.v v0, (%[p])\n"
        "vadd.vv v16, v16, v0\n"
        "addi %[p], %[p], 512\n"
        "bltu %[p], %[end], 1b\n"
        "vmv.s.x v8, zero\n"
        "vredsum.vs v8, v16, v8\n"
        "vmv.x.s %[s], v8\n"
        ".option pop\n"
        : [p] "+r"(p), [s] "=r"(s)
        : [end] "r"(end), [vl] "r"(64ul)
        : "memory",
          "v0",
          "v1",
          "v2",
          "v3",
          "v4",
          "v5",
          "v6",
          "v7",
          "v8",
          "v9",
          "v10",
          "v11",
          "v12",
          "v13",
          "v14",
          "v15",
          "v16",
          "v17",
          "v18",
          "v19",
          "v20",
          "v21",
          "v22",
          "v23"
    );
    return s;
}

/* Stores `value` into every 64-bit word of [p, p + n) with 512-byte vector stores. */
static void vfill(uintptr_t p, uint64_t n, uint64_t value) {
    const uintptr_t end = p + n;
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetvli zero, %[vl], e64, m8, ta, ma\n"
        "vmv.v.x v0, %[x]\n"
        "1:\n"
        "vse64.v v0, (%[p])\n"
        "addi %[p], %[p], 512\n"
        "bltu %[p], %[end], 1b\n"
        ".option pop\n"
        : [p] "+r"(p)
        : [end] "r"(end), [vl] "r"(64ul), [x] "r"(value)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
    );
    x280_fence_iorw();
}

static uint64_t ssum(const volatile uint64_t* p, uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n / 8; i += 8) {
        s +=
            (p[i] + p[i + 1] + p[i + 2] + p[i + 3] + p[i + 4] + p[i + 5] + p[i + 6] +
             p[i + 7]);
    }
    return s;
}

/* One phase of a size pass on this hart's quarter [q, q + n). */
static void phase(uint32_t ph, uintptr_t q, uint64_t n, uint64_t* sum) {
    switch (ph) {
        case RT_FLUSH_CLEAN:
        case RT_FLUSH_DIRTY:
            pr_flush(q, n);
            break;
        case RT_READ_COLD:
        case RT_READ_AGAIN:
            *sum = vsum(q, n);
            break;
        case RT_WRITE:
            vfill(q, n, 0x5A5A0000ull + q);
            break;
        default:
            *sum = ssum((const volatile uint64_t*)q, n);
            break;
    }
}

/* The prefix-sum job: out[c][b] = all counts of bins before b (every core) plus the
 * counts of bin b on cores before c. Hart h owns bins [h * bins / 4, ...). */
static uint32_t job(
    pr_t* p,
    const volatile rates_args_t* a,
    uint32_t* n,
    uint64_t* t_take,
    uint64_t* t_job
) {
    const uint32_t cores = (uint32_t)a->cores, bins = (uint32_t)a->bins;
    const uint32_t chunk = bins / 4, b0 = (uint32_t)p->hart * chunk;
    volatile uint32_t*
        counts = (volatile uint32_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->counts);
    volatile uint32_t* out = (
        (volatile uint32_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->out)
    );
    volatile uint32_t* totals = pr_word(p, RT_TOTALS);
    uint32_t start[RT_BINS_MAX / 4];
    uint64_t t0, t1;
    uint32_t e;
    if ((e = barrier(p, ++*n, &t0)) != 0) {
        return e;
    }
    *t_job = t0;
    /* take: drop this hart's cached lines of the counts */
    for (uint32_t c = 0; c < cores; c++) {
        pr_flush((uintptr_t)(counts + c * bins + b0), chunk * 4u);
    }
    *t_take = x280_mtime() - t0;
    uint32_t total = 0;
    for (uint32_t b = 0; b < chunk; b++) {
        start[b] = 0;
    }
    for (uint32_t c = 0; c < cores; c++) {
        for (uint32_t b = 0; b < chunk; b++) {
            start[b] += counts[c * bins + b0 + b];
        }
    }
    for (uint32_t b = 0; b < chunk; b++) {
        const uint32_t t = start[b];
        start[b] = total;
        total += t;
    }
    totals[p->hart] = total;
    if ((e = barrier(p, ++*n, &t1)) != 0) {
        return e;
    }
    x280_fence_ir();
    uint32_t base = 0;
    for (uint32_t h = 0; h < p->hart; h++) {
        base += totals[h];
    }
    for (uint32_t c = 0; c < cores; c++) {
        for (uint32_t b = 0; b < chunk; b++) {
            out[c * bins + b0 + b] = base + start[b];
            start[b] += counts[c * bins + b0 + b];
        }
    }
    /* give: write this hart's lines of the output back to DRAM */
    for (uint32_t c = 0; c < cores; c++) {
        pr_flush((uintptr_t)(out + c * bins + b0), chunk * 4u);
    }
    return 0;
}

int main(uint64_t hartid) {
    pr_t p;
    if (!pr_open(&p, hartid, sizeof(rates_args_t))) {
        x280_hart_set_result(hartid, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    pr_enable_fpv();
    const volatile rates_args_t* a = (const volatile rates_args_t*)p.lim;
    volatile uint32_t* out = pr_word(&p, PR_RESULT);
    const int sizes_ok = (a->sizes[0] % 2048 == 0) && (a->sizes[1] % 2048 == 0);
    if (!sizes_ok || a->bins == 0 || a->bins % 64 != 0 || a->bins > RT_BINS_MAX ||
        a->reps > RT_REPS_MAX || (a->region & 63) || (a->counts & 63) ||
        (a->out & 63) || a->timeout == 0 || a->timeout > 60ull * X280_MTIME_HZ) {
        if (hartid == 0) {
            out[RT_STATUS] = PR_ERR_ARGS;
        }
        x280_hart_set_result(hartid, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    p.deadline = x280_mtime() + a->timeout;
    uint32_t n = 0, e = 0;
    uint64_t t0, sum = 0;
    if (hartid == 0) {
        /* an earlier run's words; the first barrier orders this before every slot */
        for (uint32_t i = 0; i < 64; i++) {
            out[i] = 0;
        }
        x280_fence_ow();
    }
    pr_stamp(&p, 1);
    for (uint32_t pass = 0; pass < 2 && e == 0; pass++) {
        const uint64_t q = a->sizes[pass] / 4;
        if (q == 0) {
            continue;
        }
        const uintptr_t mine = X280_LOCAL_DDR_TILE_BASE + a->region + hartid * q;
        for (uint32_t ph = 0; ph < RT_PHASES && e == 0; ph++) {
            if ((e = barrier(&p, ++n, &t0)) != 0) {
                break;
            }
            phase(ph, mine, q, &sum);
            const uint32_t ticks = (uint32_t)(x280_mtime() - t0);
            volatile uint32_t* slot = out + RT_TICKS + pass * RT_PHASES + ph;
            /* each hart raises the slot to its own time: one hart at a time, in turn */
            for (uint32_t h = 0; h < 4; h++) {
                if (h == hartid && ticks > *slot) {
                    *slot = ticks;
                }
                if ((e = barrier(&p, ++n, &t0)) != 0) {
                    break;
                }
            }
            if (pass == 0 && ph == RT_READ_COLD) {
                out[RT_SUMS + hartid] = (uint32_t)sum;
            }
        }
        pr_stamp(&p, 2 + pass);
    }
    for (uint32_t r = 0; r < a->reps && e == 0; r++) {
        uint64_t take = 0, t_job = 0, t_end;
        if ((e = job(&p, a, &n, &take, &t_job)) != 0) {
            break;
        }
        if ((e = barrier(&p, ++n, &t_end)) != 0) {
            break;
        }
        if (hartid == 0) {
            out[RT_JOB + r] = (uint32_t)(t_end - t_job);
            out[RT_JOB_TAKE + r] = (uint32_t)take;
        }
    }
    pr_stamp(&p, 9);
    if (hartid == 0) {
        out[RT_STATUS] = e;
    }
    x280_hart_set_result(hartid, e ? ~(uint64_t)e : n);
    return 0;
}
