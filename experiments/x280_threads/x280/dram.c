/* DRAM visibility probe, X280 core 0: reads region A after the host and a Tensix core
 * wrote it, and writes region B for a Tensix core, without and with Flush64 (DR_*). */
#include <stdint.h>

#include "../kernels/dram_probe.h"
#include "probe.h"

typedef struct {
    uint64_t run_value;
    uint64_t a, b; /* bank addresses of the regions */
    uint64_t pages_a, pages_b, page;
    uint64_t seed_host;   /* page j of bank k of A and B: dp_word(seed_host + k, ..) */
    uint64_t seed_tensix; /* what the Tensix core writes into A */
    uint64_t seed_b1, seed_b2; /* what this image writes into B, before and after */
    uint64_t bank;             /* the bank this cluster is expected to see */
    uint64_t timeout;          /* mtime ticks for the whole run */
} dram_args_t;

/* Result words. */
enum {
    DR_STATUS = 0,    /* 0, or PR_ERR_* */
    DR_SEEN_BANK = 1, /* the bank whose host pattern page 0 of A held, or 0xFF */
    DR_A1_HOST = 2,   /* words of A equal to the host pattern, first read */
    DR_A2_TENSIX = 3, /* after the Tensix write, no invalidate: new words */
    DR_A2_HOST = 4,   /* ... words still showing the host pattern */
    DR_A3_TENSIX = 5, /* after Flush64 of A */
    DR_A3_HOST = 6,
    DR_FLUSH_TICKS = 7, /* mtime ticks of the Flush64 sweep over A */
    DR_FLUSH_LINES = 8,
    DR_B_STORE_TICKS = 9, /* ticks of the plain stores into B */
    DR_WORDS = 10,        /* words per region checked */
    DR_FAULT = 11,        /* the Flush64 store faulted (probe mode skipped it) */
    DR_STEP = 12          /* the last step reached */
};

static uint32_t count(
    volatile const uint32_t* r, uint64_t pages, uint64_t page, uint32_t seed
) {
    uint32_t n = 0;
    for (uint32_t j = 0; j < pages; j++) {
        for (uint32_t w = 0; w < page / 4; w++) {
            n += r[j * (page / 4) + w] == dp_word(seed, j, w);
        }
    }
    return n;
}

static void fill(volatile uint32_t* r, uint64_t pages, uint64_t page, uint32_t seed) {
    for (uint32_t j = 0; j < pages; j++) {
        for (uint32_t w = 0; w < page / 4; w++) {
            r[j * (page / 4) + w] = dp_word(seed, j, w);
        }
    }
    x280_fence_iorw();
}

/* The steps; returns 0 or an error, the result words filled as far as it got. */
static uint32_t probe(pr_t* p, const volatile dram_args_t* a, volatile uint32_t* out) {
    volatile uint32_t* ra = (
        (volatile uint32_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->a)
    );
    volatile uint32_t* rb = (
        (volatile uint32_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->b)
    );
    const uint64_t na = a->pages_a, nb = a->pages_b, page = a->page;
    const uint32_t host = (uint32_t)a->seed_host;
    uint32_t e;
    out[DR_WORDS] = (uint32_t)(na * page / 4);
    /* 1: what the host wrote, and which bank this is */
    out[DR_STEP] = 1;
    pr_marker(p, 1);
    out[DR_SEEN_BANK] = 0xFF;
    for (uint32_t k = 0; k < 8; k++) {
        if (ra[0] == dp_word(host + k, 0, 0) && ra[1] == dp_word(host + k, 0, 1)) {
            out[DR_SEEN_BANK] = k;
        }
    }
    const uint32_t mine = host + (uint32_t)a->bank;
    out[DR_A1_HOST] = count(ra, na, page, mine);
    pr_marker(p, 2);
    if ((e = pr_wait_go(p, 1)) != 0) {
        return e;
    }
    /* 2: the Tensix core has written A; no invalidate */
    out[DR_STEP] = 2;
    out[DR_A2_TENSIX] = count(ra, na, page, (uint32_t)a->seed_tensix);
    out[DR_A2_HOST] = count(ra, na, page, mine);
    /* 3: Flush64 over A, then read again (probe mode: a fault is skipped) */
    out[DR_STEP] = 3;
    pr_probe(p->core);
    uint64_t t = x280_mtime();
    pr_flush((uint64_t)(uintptr_t)ra, na * page);
    out[DR_FLUSH_TICKS] = (uint32_t)(x280_mtime() - t);
    out[DR_FAULT] = pr_unprobe(p->core);
    out[DR_FLUSH_LINES] = (uint32_t)(na * page / 64);
    if (out[DR_FAULT]) {
        return PR_ERR_FAULT;
    }
    out[DR_A3_TENSIX] = count(ra, na, page, (uint32_t)a->seed_tensix);
    out[DR_A3_HOST] = count(ra, na, page, mine);
    /* 4: write B with plain stores; the host copies it out (output 1) */
    out[DR_STEP] = 4;
    t = x280_mtime();
    fill(rb, nb, page, (uint32_t)a->seed_b1);
    out[DR_B_STORE_TICKS] = (uint32_t)(x280_mtime() - t);
    pr_marker(p, 4);
    if ((e = pr_wait_go(p, 2)) != 0) {
        return e;
    }
    /* 5: Flush64 over B; the host copies it out (output 2) */
    out[DR_STEP] = 5;
    pr_flush((uint64_t)(uintptr_t)rb, nb * page);
    pr_marker(p, 5);
    if ((e = pr_wait_go(p, 3)) != 0) {
        return e;
    }
    /* 6: a second pattern into B, then Flush64; the host copies it out (output 3) */
    out[DR_STEP] = 6;
    fill(rb, nb, page, (uint32_t)a->seed_b2);
    pr_flush((uint64_t)(uintptr_t)rb, nb * page);
    pr_marker(p, 6);
    if ((e = pr_wait_go(p, 4)) != 0) {
        return e;
    }
    out[DR_STEP] = 7;
    return 0;
}

int main(uint64_t core_id) {
    if (core_id != 0) {
        return 0;
    }
    pr_t p;
    if (!pr_open(&p, core_id, sizeof(dram_args_t))) {
        x280_hart_set_result(core_id, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    const volatile dram_args_t* a = (const volatile dram_args_t*)p.lim;
    volatile uint32_t* out = pr_word(&p, PR_RESULT);
    for (uint32_t i = 0; i < 16; i++) {
        out[i] = 0;
    }
    if (
        a->page == 0 || a->page % 64 != 0 || a->pages_a == 0 || a->pages_b == 0 ||
        (a->a & 63) || (a->b & 63) || a->bank > 7 || a->timeout == 0 ||
        a->timeout > 30ull * X280_MTIME_HZ
    ) {
        out[DR_STATUS] = PR_ERR_ARGS;
        x280_hart_set_result(core_id, ~(uint64_t)PR_ERR_ARGS);
        return 0;
    }
    p.deadline = x280_mtime() + a->timeout;
    const uint32_t e = probe(&p, a, out);
    out[DR_STATUS] = e;
    pr_marker(&p, 9);
    x280_hart_set_result(core_id, e ? ~(uint64_t)e : 0);
    return 0;
}
