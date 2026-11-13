/* Work-cluster probe thread (kmx): sums, DRAM rates, pushes to another bank through a
 * NoC window, wakes between clusters, kmx_par round trips. Word 0 picks the mode. */
#include <stdint.h>

#include "kmx.h"
#include "noc.h"
#include "rt/x280_hw.h"
#include "x280_address_map.h"

enum { M_SUM = 1, M_RATE = 2, M_PUSH = 3, M_PING = 4, M_PAR = 5 };

/* Words common to every mode. */
enum { W_MODE = 0, W_GO = 1 };

/* A go line the host writes after every cluster started (word 0 = words[W_GO]). */
#define GO_LIM 0x480u
#define WIN0 40u /* core h pushes through window WIN0 + h; WIN0 + 4 the ping window */
#define SPINS 256u
#define CODE_STOPPED 9u

typedef struct {
    uint64_t addr, bytes, seed, out[4];
    uint64_t noc, dest; /* M_PUSH: tile x | y << 16, bank address there */
} job_t;

static uint32_t* lim_word(uint32_t at) {
    return (uint32_t*)((uintptr_t)kmx_args_get() + at);
}

/* Waits for the host's go word; 0 or CODE_STOPPED. */
static uint32_t wait_go(void) {
    volatile uint32_t* go = lim_word(GO_LIM);
    const uint32_t want = kmx_args_get()->words[W_GO];
    for (uint32_t s = 0;; s++) {
        __asm__ volatile("fence ir, ir" ::: "memory");
        if (*go == want) {
            return 0;
        }
        if (s % SPINS == 0 && kmx_stop_requested()) {
            return CODE_STOPPED;
        }
    }
}

/* Core h's quarter of [addr, addr + bytes) in 512-byte units. */
static void quarter(const job_t* j, int h, uint64_t* p, uint64_t* n) {
    const uint64_t units = j->bytes / 512, per = (units + 3) / 4;
    const uint64_t first = (uint64_t)h * per;
    const uint64_t k = first >= units ? 0 : (units - first < per ? units - first : per);
    *p = j->addr + first * 512;
    *n = k * 512;
}

static uint64_t vsum(uintptr_t p, uint64_t n) {
    if (n == 0) {
        return 0;
    }
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

/* Stores word i of the pattern, seed ^ i * K, over [dst, dst + n) where i counts
 * 64-bit words from `first`; 512 bytes per store. */
static void vfill(uintptr_t dst, uint64_t n, uint64_t first, uint64_t seed) {
    const uintptr_t end = dst + n;
    const uint64_t k = 0x9E3779B97F4A7C15ull;
    while (dst < end) {
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetvli zero, %[vl], e64, m8, ta, ma\n"
            "vid.v v0\n"
            "vadd.vx v0, v0, %[i]\n"
            "vmul.vx v0, v0, %[k]\n"
            "vxor.vx v0, v0, %[s]\n"
            "vse64.v v0, (%[d])\n"
            ".option pop\n"
            :
            : [vl] "r"(64ul), [i] "r"(first), [k] "r"(k), [s] "r"(seed), [d] "r"(dst)
            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
        );
        dst += 512;
        first += 64;
    }
}

static void par_sum(int h, void* arg) {
    job_t* j = (job_t*)arg;
    uint64_t p, n;
    quarter(j, h, &p, &n);
    j->out[h] = vsum((uintptr_t)p, n);
}

static void par_fill(int h, void* arg) {
    job_t* j = (job_t*)arg;
    uint64_t p, n;
    quarter(j, h, &p, &n);
    vfill((uintptr_t)p, n, (p - j->addr) / 8, j->seed);
}

static void par_none(int h, void* arg) {
    (void)h;
    (void)arg;
}

/* Core h pushes its quarter of the pattern to [dest, dest + bytes) of tile `noc`
 * through its own window, re-aimed at each 2 MiB; out[h] = 1 if a window failed. */
static void par_push(int h, void* arg) {
    job_t* j = (job_t*)arg;
    const job_t at = {.addr = j->dest, .bytes = j->bytes};
    uint64_t p, n;
    quarter(&at, h, &p, &n);
    const uint32_t x = (uint32_t)(j->noc & 0xFFFFu), y = (uint32_t)(j->noc >> 16);
    j->out[h] = 0;
    while (n != 0) {
        const uint64_t room = NOC_2M_WINDOW_STRIDE - (p & (NOC_2M_WINDOW_STRIDE - 1));
        const uint64_t k = n < room ? n : room;
        noc_tlb_2m_t prog;
        volatile uint8_t* w = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                WIN0 + (uint32_t)h, x, y, p, NOC_ORDER_POSTED, &prog
            )
        );
        if (w == 0) {
            j->out[h] = 1;
            return;
        }
        vfill((uintptr_t)w, k, (p - j->dest) / 8, j->seed);
        x280_fence_ow();
        p += k;
        n -= k;
    }
    noc_clear_tlb_2m(WIN0 + (uint32_t)h);
}

/* Core 0 alone: 8-byte loads of [dest, dest + bytes) of tile `noc` (System Port). */
static uint64_t pull_sum(const job_t* j, uint32_t* bad) {
    const uint32_t x = (uint32_t)(j->noc & 0xFFFFu), y = (uint32_t)(j->noc >> 16);
    uint64_t s = 0, p = j->dest, n = j->bytes;
    while (n != 0) {
        const uint64_t room = NOC_2M_WINDOW_STRIDE - (p & (NOC_2M_WINDOW_STRIDE - 1));
        const uint64_t k = n < room ? n : room;
        noc_tlb_2m_t prog;
        volatile uint64_t* w = (
            (volatile uint64_t*)noc_configure_tlb_2m_checked(
                WIN0, x, y, p, NOC_ORDER_DEFAULT, &prog
            )
        );
        if (w == 0) {
            *bad = 1;
            return 0;
        }
        for (uint64_t i = 0; i < k / 8; i++) {
            s += w[i];
        }
        p += k;
        n -= k;
        if (kmx_stop_requested()) {
            break;
        }
    }
    noc_clear_tlb_2m(WIN0);
    return s;
}

static void flush_range(uint64_t addr, uint64_t bytes) {
    volatile uint64_t* reg = (volatile uint64_t*)(uintptr_t)0x02010200ull;
    __asm__ volatile("fence iorw, iorw" ::: "memory");
    for (uint64_t a = addr & ~63ull; a < addr + bytes; a += 64) {
        __asm__ volatile(".option push\n.option norvc\nsd %0, 0(%1)\n.option pop\n"
                         :
                         : "r"(a), "r"(reg)
                         : "memory");
    }
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

/* Ping words: role, rounds, kind, my flag and the peer's (bank address, LIM offset for
 * kind 2), the peer's tile (kinds 1, 2), exchange address, bytes, seed (A fills it). */
enum { P_ROLE = 2, P_ROUNDS, P_KIND, P_MINE, P_PEER, P_NOC, P_XADDR, P_XBYTES, P_SEED };

enum { K_SHARED = 0, K_PUSH_DRAM = 1, K_PUSH_LIM = 2 };

typedef struct {
    uint32_t kind;
    volatile uint32_t* mine;
    volatile uint32_t* peer;
    uint64_t peer_line; /* K_SHARED: the peer's line as this cluster sees it */
} ping_t;

static uint32_t ping_open(const kmx_args* a, ping_t* q) {
    q->kind = a->words[P_KIND];
    const uint32_t x = a->words[P_NOC] & 0xFFFFu, y = a->words[P_NOC] >> 16;
    if (q->kind == K_PUSH_LIM) {
        const uint64_t base = (uint64_t)(uintptr_t)kmx_args_get();
        q->mine = (volatile uint32_t*)(uintptr_t)(base + a->words[P_MINE]);
        noc_tlb_2m_t prog;
        volatile uint8_t* w = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                WIN0 + 4, x, y, X280_LIM_BASE + 0x0u, NOC_ORDER_POSTED, &prog
            )
        );
        if (w == 0) {
            return 1;
        }
        /* the peer's LIM block is at the same available-LIM base as this one's */
        q->peer = (volatile uint32_t*)(w + (base - X280_LIM_BASE) + a->words[P_PEER]);
        return 0;
    }
    q->mine = (
        (volatile uint32_t*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->words[P_MINE])
    );
    if (q->kind == K_SHARED) {
        q->peer_line = X280_LOCAL_DDR_TILE_BASE + a->words[P_PEER];
        q->peer = (volatile uint32_t*)(uintptr_t)q->peer_line;
        return 0;
    }
    noc_tlb_2m_t prog;
    volatile uint8_t* w = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            WIN0 + 4, x, y, a->words[P_PEER], NOC_ORDER_POSTED, &prog
        )
    );
    if (w == 0) {
        return 1;
    }
    q->peer = (volatile uint32_t*)w;
    return 0;
}

static void ping_send(ping_t* q, uint32_t v) {
    if (q->kind == K_SHARED) {
        q->peer[0] = v;
        flush_range(q->peer_line, 64);
        return;
    }
    q->peer[0] = v;
    x280_fence_ow();
}

static uint32_t ping_wait(ping_t* q, uint32_t v) {
    for (uint32_t s = 1;; s++) {
        if (q->kind != K_PUSH_LIM) {
            flush_range((uint64_t)(uintptr_t)q->mine, 64);
        } else {
            __asm__ volatile("fence ir, ir" ::: "memory");
        }
        if (q->mine[0] == v) {
            return 0;
        }
        if (s % SPINS == 0 && kmx_stop_requested()) {
            return CODE_STOPPED;
        }
    }
}

static uint32_t ping(const kmx_args* a, uint32_t* r) {
    ping_t q;
    if (ping_open(a, &q)) {
        return 3;
    }
    const uint32_t role = a->words[P_ROLE], rounds = a->words[P_ROUNDS];
    uint32_t e = wait_go();
    job_t x = {.addr = a->words[P_XADDR], .bytes = a->words[P_XBYTES]};
    x.seed = a->words[P_SEED] | (uint64_t)a->words[P_SEED] << 32;
    if (e == 0 && role == 0 && x.bytes) {
        /* A: the exchange region, then wake 0x80000000 */
        if (q.kind == K_SHARED) {
            x.addr += X280_LOCAL_DDR_TILE_BASE;
            e = kmx_par(par_fill, &x) == KMX_OK ? 0 : CODE_STOPPED;
            flush_range(x.addr, x.bytes);
        } else {
            x.dest = x.addr;
            x.noc = a->words[P_NOC];
            e = kmx_par(par_push, &x) == KMX_OK ? 0 : CODE_STOPPED;
            e = e ? e : (uint32_t)(x.out[0] | x.out[1] | x.out[2] | x.out[3]) * 4u;
        }
        kmx_progress(2);
        if (e == 0) {
            ping_send(&q, 0x80000000u);
            e = ping_wait(&q, 0x80000000u); /* B's answer: it saw the region */
        }
    }
    if (e == 0 && role == 1 && x.bytes) {
        e = ping_wait(&q, 0x80000000u);
        kmx_progress(3);
        x.addr += X280_LOCAL_DDR_TILE_BASE;
        flush_range(x.addr, x.bytes);
        if (e == 0) {
            e = kmx_par(par_sum, &x) == KMX_OK ? 0 : CODE_STOPPED;
            const uint64_t s = x.out[0] + x.out[1] + x.out[2] + x.out[3];
            r[2] = (uint32_t)s;
            r[3] = (uint32_t)(s >> 32);
            ping_send(&q, 0x80000000u);
        }
    }
    kmx_progress(4);
    const uint64_t t0 = kmx_time();
    for (uint32_t i = 1; e == 0 && i <= rounds; i++) {
        kmx_progress(0x100 + i);
        if (role == 0) {
            ping_send(&q, i);
            e = ping_wait(&q, i);
        } else {
            e = ping_wait(&q, i);
            ping_send(&q, i);
        }
    }
    r[0] = (uint32_t)(kmx_time() - t0);
    r[1] = rounds;
    if (q.kind != K_SHARED) {
        noc_clear_tlb_2m(WIN0 + 4);
    }
    return e;
}

static uint32_t r[48] __attribute__((section(".data"))) = {1};

int kmx_main(int core) {
    (void)core;
    const kmx_args* a = kmx_args_get();
    uint32_t code = 0, n = 0;
    uint64_t bytes;
    void* data = kmx_region(0, &bytes);
    const uint32_t mode = a->words[W_MODE];
    if (mode == M_SUM) {
        job_t j = {.addr = (uint64_t)(uintptr_t)data, .bytes = bytes & ~511ull};
        const uint64_t t0 = kmx_time();
        kmx_invalidate(0);
        code = kmx_par(par_sum, &j) == KMX_OK ? 0 : CODE_STOPPED;
        const uint64_t s = j.out[0] + j.out[1] + j.out[2] + j.out[3];
        r[0] = (uint32_t)s;
        r[1] = (uint32_t)(s >> 32);
        r[2] = (uint32_t)(kmx_time() - t0);
        r[3] = (uint32_t)kmx_cluster();
        n = 4;
    } else if (mode == M_RATE) {
        /* words: reps; per rep: invalidate, read, write, flush (ticks); r[32..33] last
         * sum */
        const uint32_t reps = a->words[2] < 7 ? a->words[2] : 7;
        job_t j = {.addr = (uint64_t)(uintptr_t)data, .bytes = bytes & ~511ull};
        j.seed = 0x1234;
        code = wait_go();
        for (uint32_t i = 0; code == 0 && i < reps; i++) {
            uint64_t t = kmx_time();
            kmx_invalidate(0);
            r[4 * i] = (uint32_t)(kmx_time() - t);
            t = kmx_time();
            code = kmx_par(par_sum, &j) == KMX_OK ? 0 : CODE_STOPPED;
            r[4 * i + 1] = (uint32_t)(kmx_time() - t);
            t = kmx_time();
            code = code ? code : (kmx_par(par_fill, &j) == KMX_OK ? 0 : CODE_STOPPED);
            r[4 * i + 2] = (uint32_t)(kmx_time() - t);
            t = kmx_time();
            kmx_flush(0);
            r[4 * i + 3] = (uint32_t)(kmx_time() - t);
        }
        const uint64_t s = j.out[0] + j.out[1] + j.out[2] + j.out[3];
        r[32] = (uint32_t)s;
        r[33] = (uint32_t)(s >> 32);
        n = 34;
    } else if (mode == M_PUSH) {
        /* words: noc, dest, bytes, seed, pull bytes: push ticks, pull ticks, pull sum
         */
        job_t j = {.noc = a->words[2], .dest = a->words[3], .bytes = a->words[4]};
        j.seed = a->words[5] | (uint64_t)a->words[5] << 32;
        code = wait_go();
        uint64_t t = kmx_time();
        if (code == 0) {
            code = kmx_par(par_push, &j) == KMX_OK ? 0 : CODE_STOPPED;
            code =
                (code ? code
                      : (uint32_t)(j.out[0] | j.out[1] | j.out[2] | j.out[3]) * 4u);
        }
        r[0] = (uint32_t)(kmx_time() - t);
        uint32_t bad = 0;
        j.bytes = a->words[6];
        t = kmx_time();
        const uint64_t s = code == 0 ? pull_sum(&j, &bad) : 0;
        r[1] = (uint32_t)(kmx_time() - t);
        r[2] = (uint32_t)s;
        r[3] = (uint32_t)(s >> 32);
        code = code ? code : bad * 5u;
        n = 4;
    } else if (mode == M_PING) {
        code = ping(a, r);
        n = 4;
    } else if (mode == M_PAR) {
        const uint32_t calls = a->words[2];
        const uint64_t t = kmx_time();
        for (uint32_t i = 0; code == 0 && i < calls; i++) {
            code = kmx_par(par_none, 0) == KMX_OK ? 0 : CODE_STOPPED;
        }
        r[0] = (uint32_t)(kmx_time() - t);
        r[1] = calls;
        n = 2;
    } else {
        code = 1;
    }
    kmx_result(code, r, n);
    return (int)code;
}
