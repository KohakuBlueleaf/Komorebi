/* The thread: each X280 core decides its Tensix cores' bodies from its schedule; by
 * mode the core reads it, or the X280 core stores it (or a constant) into the slot. */
#include <stdint.h>

#include "kmx.h"
#include "noc.h"
#include "remote_x280.h"
#include "rt/x280_hw.h"

#define CODE_BAD 1u  /* words or regions out of range */
#define CODE_LATE 2u /* the command was late or stopped, or the ring stayed full */
#define CODE_FAIL 3u /* a core replied with an error, or a store's window failed */
#define FULL_TICKS KMX_TICKS_PER_SECOND
/* The window an X280 core stores through: after the ports' windows, one per core. */
#define RL_WINDOW (KMX_PORT_WINDOW + KMX_PORTS_CORES)
#define REG_CFG 0
#define REG_BODIES 1
#define REG_OUT 2

/* Result words. */
enum { R_SWAPS, R_FAILS, R_TICKS_LO, R_TICKS_HI, R_COUNT };

static uint32_t g_fails __attribute__((section(".data"))) = 0;
/* Per X280 core: the cores it serves, each one's next swap and when it was decided. */
static uint8_t g_mine[4][KMX_PORTS_CORES] __attribute__((section(".data"))) = {{0}};
static uint32_t g_k[4][KMX_PORTS_CORES] __attribute__((section(".data"))) = {{0}};
static uint64_t g_t0[4][KMX_PORTS_CORES] __attribute__((section(".data"))) = {{0}};

static uint32_t word(int i) {
    return kmx_args_get()->words[i];
}

static uint32_t* region(int id) {
    uint64_t bytes;
    return (uint32_t*)kmx_region(id, &bytes);
}

/* 64-byte multiples of `src` stored through a window, 512 bytes per vector store. */
static void copy_out(volatile uint8_t* dst, const uint8_t* src, uint64_t n) {
    while (n != 0) {
        uint64_t vl;
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetvli %0, %1, e64, m8, ta, ma\n"
            "vle64.v v0, (%2)\n"
            "vse64.v v0, (%3)\n"
            ".option pop\n"
            : "=&r"(vl)
            : "r"(n / 8), "r"(src), "r"(dst)
            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
        );
        src += vl * 8;
        dst += vl * 8;
        n -= vl * 8;
    }
}

/* [src, src + n) (64 B multiples) to L1 `at` of tile `xy`: posted stores, then the
 * last line again non-posted (acknowledged after them on one path). 0, or 1. */
static int store(
    uint32_t win, uint32_t xy, uint32_t at, const uint8_t* src, uint32_t n
) {
    const uint32_t x = xy & 0xFFFFu, y = xy >> 16;
    noc_tlb_2m_t prog;
    volatile uint8_t* w = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            win, x, y, at, NOC_ORDER_POSTED, &prog
        )
    );
    if (w == 0) {
        return 1;
    }
    copy_out(w, src, n);
    x280_fence_ow();
    volatile uint32_t* d = (
        (volatile uint32_t*)noc_configure_tlb_2m_checked(
            win, x, y, at + n - 64u, NOC_ORDER_DEFAULT, &prog
        )
    );
    if (d == 0) {
        return 1;
    }
    const volatile uint32_t* s = (const volatile uint32_t*)(src + n - 64u);
    for (int i = 0; i < 16; i++) {
        d[i] = s[i];
    }
    x280_fence_ow();
    noc_clear_tlb_2m(win);
    return 0;
}

/* The template's constant `v` as its lui / addi pair (fields masked), stored as two
 * words through a non-posted window at L1 address `at` of tile `xy`. 0, or 1. */
static int patch(uint32_t win, uint32_t xy, uint32_t at, uint32_t rd, uint32_t v) {
    const uint32_t hi = ((v + 0x800u) >> 12) & 0xFFFFFu, lo = v & 0xFFFu;
    rd &= 0x1Fu;
    const uint32_t lui = hi << 12 | rd << 7 | 0x37u;
    const uint32_t addi = lo << 20 | rd << 15 | rd << 7 | 0x13u;
    noc_tlb_2m_t prog;
    volatile uint32_t* d = (
        (volatile uint32_t*)noc_configure_tlb_2m_checked(
            win, xy & 0xFFFFu, xy >> 16, at, NOC_ORDER_DEFAULT, &prog
        )
    );
    if (d == 0) {
        return 1;
    }
    d[0] = lui;
    d[1] = addi;
    x280_fence_ow();
    noc_clear_tlb_2m(win);
    return 0;
}

/* Swap k of core c: the body from the X280 core's schedule, the bytes moved by mode,
 * the message; body, argument, store ticks into its line. 0, or 1: a store failed. */
static int submit(int cluster, int cpu, int c, uint32_t k, uint32_t* line) {
    const uint32_t* cfg = region(REG_CFG);
    const uint32_t mode = word(RL_W_MODE), slot = word(RL_W_SLOT);
    const uint32_t b = (
        word(RL_W_FIRST) +
        cfg[RL_CFG_SCHED + (4u * cluster + cpu) * RL_SCHED + k % RL_SCHED]
    );
    const uint32_t bytes = cfg[RL_CFG_BYTES + b], xy = cfg[RL_CFG_NOC + c];
    const uint8_t* src = (const uint8_t*)region(REG_BODIES) + cfg[RL_CFG_OFFSET + b];
    const uint32_t win = RL_WINDOW + (uint32_t)cpu, arg = RL_ARG(c, k);
    line[RL_L_BODY] = b;
    line[RL_L_ARG] = arg;
    const uint64_t t0 = kmx_time();
    int bad = 0;
    if (mode == RL_MODE_STORE || (mode == RL_MODE_PATCH && k == 0)) {
        bad = store(win, xy, slot, src, bytes);
    }
    if (mode == RL_MODE_PATCH && !bad) {
        bad = (
            patch(win, xy, slot + word(RL_W_MARK_AT), word(RL_W_MARK_RD), RL_VALUE(k))
        );
    }
    line[RL_L_STORE] = (uint32_t)(kmx_time() - t0);
    if (bad) {
        return 1;
    }
    if (mode == RL_MODE_SELF) {
        /* copy c % 8 of body b: page b * 8 + c % 8 of the images */
        const uint32_t p = b * 8u + (uint32_t)c % 8u;
        const uint32_t at = word(RL_W_IMG_BASE) + (p / 8u) * word(RL_W_IMG_PAGE);
        kmx_port_tell(c, at, RL_LOAD | (p % 8u) << 24 | bytes, arg);
    } else {
        kmx_port_tell(c, 0, bytes, arg);
    }
    return 0;
}

/* One X280 core: its cores' swaps, each next one decided when the last one replied;
 * then RL_END to each. */
static void serve(int cluster, int cpu, uint64_t arg) {
    (void)arg;
    const uint32_t swaps = word(RL_W_SWAPS), cores = word(RL_W_CORES);
    uint32_t* out = region(REG_OUT);
    uint8_t* mine = g_mine[cpu];
    uint32_t* k = g_k[cpu];
    uint64_t* t0 = g_t0[cpu];
    int n = 0;
    for (int c = 0; c < (int)cores; c++) {
        if (kmx_port_owner(c) == cpu) {
            mine[n++] = (uint8_t)c;
        }
    }
    int left = n;
    for (int i = 0; i < n; i++) {
        k[i] = 0;
        t0[i] = kmx_time();
        if (submit(cluster, cpu, mine[i], 0, out + RL_LINE * swaps * mine[i])) {
            __atomic_fetch_add(&g_fails, 1u, __ATOMIC_RELAXED);
            kmx_port_tell(mine[i], 0, RL_END, 0);
            k[i] = swaps;
            left--;
        }
    }
    while (left > 0 && kmx_port_open()) {
        for (int i = 0; i < n; i++) {
            const int c = mine[i];
            kmx_msg m;
            if (k[i] >= swaps || kmx_port_told(c, &m) != KMX_OK) {
                continue;
            }
            uint32_t* l = out + RL_LINE * (swaps * c + k[i]);
            l[RL_L_TICKS] = (uint32_t)(kmx_time() - t0[i]);
            l[RL_L_RESULT] = m.w[0];
            l[RL_L_CALLS] = m.w[1];
            l[RL_L_CYCLES] = m.w[2];
            l[RL_L_WHO] = 1u + ((uint32_t)cluster << 2 | (uint32_t)cpu);
            if (m.w[1] == RL_FAIL) {
                __atomic_fetch_add(&g_fails, 1u, __ATOMIC_RELAXED);
                k[i] = swaps;
            } else {
                k[i]++;
            }
            if (k[i] < swaps) {
                t0[i] = kmx_time();
                if (submit(cluster, cpu, c, k[i], out + RL_LINE * (swaps * c + k[i]))) {
                    __atomic_fetch_add(&g_fails, 1u, __ATOMIC_RELAXED);
                    k[i] = swaps;
                }
            }
            if (k[i] >= swaps) {
                kmx_port_tell(c, 0, RL_END, 0);
                left--;
            }
        }
    }
}

/* X280 core 0 of every cluster: the lines into the bank, for the host. */
static void flush_out(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (cpu == 0) {
        kmx_flush(REG_OUT);
    }
}

/* X280 core 0 of every cluster: the host's regions invalidated before they are read. */
static void invalidate_in(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (cpu == 0) {
        kmx_invalidate(REG_CFG);
        kmx_invalidate(REG_BODIES);
    }
}

int kmx_main(int cpu) {
    (void)cpu;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
    const kmx_args* a = kmx_args_get();
    uint64_t cfg_bytes = 0, out_bytes = 0, bodies_bytes = 0;
    const uint32_t cores = a->nwords >= RL_WORDS ? a->words[RL_W_CORES] : 0;
    if (
        a->nwords < RL_WORDS || kmx_region(REG_CFG, &cfg_bytes) == 0 ||
        cfg_bytes < 4u * RL_CFG_WORDS || kmx_region(REG_BODIES, &bodies_bytes) == 0 ||
        kmx_region(REG_OUT, &out_bytes) == 0 || cores == 0 ||
        cores > (uint32_t)kmx_port_cores() ||
        (uint64_t)cores * a->words[RL_W_SWAPS] > RL_LINES_MAX ||
        out_bytes < 4u * RL_LINE * RL_LINES_MAX
    ) {
        return CODE_BAD;
    }
    if (kmx_par_all(invalidate_in, 0) != KMX_OK) {
        return CODE_LATE;
    }
    km_command_t r;
    r.seq = r.land1 = r.land2 = r.land3 = 0;
    r.hdr = RL_HANDLER_SWAP;
    r.group = 0 | cores << 16;
    r.a0 = r.a1 = r.a2 = r.a3 = r.a4 = r.a5 = r.a6 = r.a7 = r.a8 = r.a9 = 0;
    kmx_event ev = {0, 0, 0, 0};
    const uint64_t t0 = kmx_time(), until = t0 + FULL_TICKS;
    int e = kmx_port_run(&r, serve, 0, &ev);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_port_run(&r, serve, 0, &ev);
    }
    const uint64_t ticks = kmx_time() - t0;
    if (kmx_par_all(flush_out, 0) != KMX_OK) {
        return CODE_LATE;
    }
    uint32_t w[R_COUNT] = {0, g_fails, (uint32_t)ticks, (uint32_t)(ticks >> 32)};
    w[R_SWAPS] = ev.kind == KM_EV_DONE ? cores * a->words[RL_W_SWAPS] : 0;
    kmx_result(e != KMX_OK ? CODE_LATE : g_fails ? CODE_FAIL : 0, w, R_COUNT);
    return 0;
}
