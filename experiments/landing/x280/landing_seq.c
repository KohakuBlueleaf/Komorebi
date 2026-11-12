/* Landing experiment: X280 record issuer (hart 0). For record i = 1..n it writes the
 * 64 B record into slot i % depth (slot 0 at depth 1) of every core's control block
 * with one vector store per multicast panel: protocol 0 (A): words 0, 4, 8, 12 are the
 * echo word tag << 16 | i, the rest payload protocol 1 (B): 16 payload words, then
 * (after a fence) the sequence word tag << 16 | i at LANDING_SEQ_OFF through the same
 * window Pacing: depth 1 waits for every column leader's ack of i before i + 1; depth D
 * > 1 issues i once every ack is >= i - D. Result: mtime ticks from record 1 issued to
 * the last ack, or ~k on error. */
#include <stdint.h>

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_hart_protocol.h"

#include "../kernels/landing.h"

#define TTX_READY 0xFFFFFFFFu

/* Arguments written by the host (landing_x280.cpp Plan::start) at the start of the
 * available LIM. Must match LandingArgs there. */
typedef struct {
    uint64_t run_value;
    uint64_t mailbox;
    uint64_t acks;
    uint64_t n;
    uint64_t k;
    uint64_t nrect;
    uint32_t rect[2][4];
    uint64_t ack_stride;
    uint64_t ordering;
    uint64_t proto;
    uint64_t depth;
    uint64_t tag;
} landing_args_t;

static inline void store_line(volatile uint8_t* dst, const uint32_t* line) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 16, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(line), "r"(dst)
        : "memory", "v8"
    );
}

typedef struct {
    uint64_t hart, run;
} ctx_t;

static int stopped(const ctx_t* c, uint32_t* spins) {
    if (++*spins == 4096u) {
        *spins = 0;
        return x280_hart_stop_requested(c->hart, c->run);
    }
    return 0;
}

/* All k ack slots reach value (READY only when value == READY). */
static int wait_all(
    const ctx_t* c,
    volatile const uint32_t* acks,
    uint64_t k,
    uint64_t stride_words,
    uint32_t value
) {
    uint32_t spins = 0;
    uint64_t from = 0;
    for (;;) {
        x280_fence_ir();
        while (from < k) {
            const uint32_t v = acks[from * stride_words];
            if (value == TTX_READY ? v != TTX_READY
                                   : (v == TTX_READY || (int32_t)(v - value) < 0)) {
                break;
            }
            from++;
        }
        if (from == k) {
            return 1;
        }
        if (stopped(c, &spins)) {
            return 0;
        }
    }
}

int main(uint64_t hartid) {
    if (hartid != 0) {
        return 0;
    }
    const ctx_t ctx = {hartid, x280_hart_control(hartid)->run_value};
    const volatile landing_args_t* a = (
        (const volatile landing_args_t*) x280_hart_args(hartid)
    );
    if (a == 0 || a->run_value != ctx.run || a->nrect < 1 || a->nrect > 2 ||
        a->ordering > 2 || a->proto > 1 || a->depth < 1 || a->depth > LANDING_SLOTS) {
        x280_hart_set_result(hartid, ~0ull);
        return 0;
    }
    volatile uint32_t* mb = (volatile uint32_t*)(uintptr_t)a->mailbox;
    volatile uint32_t* acks = (volatile uint32_t*)(uintptr_t)a->acks;
    const uint64_t n = a->n, k = a->k, nrect = a->nrect, sw = a->ack_stride / 4;
    const uint32_t depth = (uint32_t)a->depth, tag = (uint32_t)a->tag & 0xFFFFu;
    const int proto_b = a->proto == 1;

    if (!x280_hart_wait_word_or_stop(hartid, ctx.run, &mb[0], 1u)) {
        return 0;
    }
    if (!wait_all(&ctx, acks, k, sw, TTX_READY)) {
        return 0;
    }
    const uint32_t ctl_addr = mb[1];
    if (ctl_addr & 63u) {
        x280_hart_set_result(hartid, ~4ull);
        return 0;
    }
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    volatile uint8_t* ctl[2];
    for (uint64_t r = 0; r < nrect; r++) {
        noc_tlb_2m_t programmed;
        const uint32_t* q = (const uint32_t*)a->rect[r];
        const int mcast = !(q[0] == q[2] && q[1] == q[3]);
        volatile void* win = (
            noc_configure_tlb_2m_mcast(
                (uint32_t)r,
                q[0],
                q[1],
                q[2],
                q[3],
                0,
                (noc_ordering_t)a->ordering,
                mcast,
                0u,
                &programmed
            )
        );
        if (win == 0 || noc_verify_tlb_2m((uint32_t)r, &programmed) != 0) {
            x280_hart_set_result(hartid, ~2ull);
            return 0;
        }
        ctl[r] = (volatile uint8_t*)win + ctl_addr;
    }

    uint32_t line[16] __attribute__((aligned(64)));
    int ok = 1;
    const uint64_t t0 = x280_mtime();
    for (uint32_t i = 1; i <= n && ok; i++) {
        if (depth > 1 && i > depth) {
            ok = wait_all(&ctx, acks, k, sw, i - depth);
            if (!ok) {
                break;
            }
        }
        const uint32_t echo = landing_echo(i, tag);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = (!proto_b && (w & 3u) == 0) ? echo : landing_word(i, w, tag);
        }
        const uint32_t off = (depth > 1 ? i % depth : 0) * LANDING_SLOT_BYTES;
        for (uint64_t r = 0; r < nrect; r++) {
            store_line(ctl[r] + off, line);
        }
        if (proto_b) {
            __asm__ volatile("fence ow, ow" ::: "memory");
            for (uint64_t r = 0; r < nrect; r++) {
                *(volatile uint32_t*)(ctl[r] + LANDING_SEQ_OFF) = echo;
            }
        }
        __asm__ volatile("fence ow, ow" ::: "memory");
        mb[4] = i; /* last issued record, for post-mortem */
        if (depth == 1) {
            ok = wait_all(&ctx, acks, k, sw, i);
        }
    }
    if (ok && depth > 1) {
        ok = wait_all(&ctx, acks, k, sw, (uint32_t)n);
    }
    const uint64_t t1 = x280_mtime();
    __asm__ volatile("fence ow, ow" ::: "memory");
    for (uint64_t r = 0; r < nrect; r++) {
        noc_clear_tlb_2m((uint32_t)r);
    }
    x280_hart_set_result(hartid, ok ? (t1 - t0) : ~3ull);
    return 0;
}
