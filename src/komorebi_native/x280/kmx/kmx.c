/* kmx start-up and calls: every hart checks the argument block; hart 0 checks the
 * bank and runs kmx_main; harts 1 to 3 run what kmx_spawn gives them until it ends. */
#include <stdint.h>

#include "kmx.h"
#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"
#include "x280_hart_protocol.h"

/* The cluster's L2 cache controller: Flush64 writes back and invalidates the line
 * that holds the physical address stored into it. */
#define KMX_CCACHE_FLUSH64 0x02010200ull
#define KMX_POLL_EVERY 256u
/* kmx_port_open looks for the event every KMX_PORT_LOOK calls on the lead's X280 core
 * 0, for a stop or the deadline every KMX_PORT_CHECK calls. */
#define KMX_PORT_LOOK 4u
#define KMX_PORT_CHECK 16u
/* Hart h's L2 prefetcher (BASIC_CONTROL, USER_CONTROL) and the documented values,
 * set unless the host asks to keep them: scalar code that walks memory runs faster. */
#define KMX_L2PF_BASE 0x02030000ull
#define KMX_L2PF_STRIDE 0x2000ull
#define KMX_L2PF_BASIC 0x15811u
#define KMX_L2PF_USER 0x38c84eu
#define KMX_UNSET 0xFFFFFFFFu

static uint64_t g_deadline[4] __attribute__((section(".data"))) = {1, 1, 1, 1};
static int must_end(uint64_t h);
static void flush(uint64_t addr, uint64_t bytes);
static inline uint64_t hart_id(void);

#define KMX_RING_FENCE_W() x280_fence_ow()
#define KMX_RING_FENCE_R() x280_fence_ir()
#define KMX_RING_SWEEP(p, n) flush((p), (n))
#define KMX_RING_NOW() x280_mtime()
#define KMX_RING_END() must_end(0)
#include "kmx_ring.h"

/* The ring, hart 0's (a .data global: the image has no .bss). */
static kmx_ring_t g_ring __attribute__((section(".data"))) = {.next = 1};

/* Image globals live in .data (the image has no .bss): non-zero initialisers. */
static uint64_t g_start __attribute__((section(".data"))) = 1;
static uint32_t g_code __attribute__((section(".data"))) = KMX_UNSET;
static uint32_t g_in_par __attribute__((section(".data"))) = 0; /* hart 0 in kmx_par */
/* kmx_par_all: the call number, whether every other cluster said it serves, and
 * each other cluster's LIM block through its window (0: none). */
static uint32_t g_all_gen __attribute__((section(".data"))) = 0;
static uint32_t g_all_ready __attribute__((section(".data"))) = 0;
static volatile uint8_t* g_peer[4] __attribute__((section(".data"))) = {0, 0, 0, 0};

static inline uint64_t hart_id(void) {
    uint64_t h;
    __asm__ volatile("csrr %0, mhartid" : "=r"(h));
    return h;
}

static inline volatile uint8_t* lim(void) {
    return (volatile uint8_t*)(uintptr_t)x280_hart_args(hart_id());
}

static inline volatile kmx_result_block* result(void) {
    return (volatile kmx_result_block*)(lim() + KMX_LIM_RESULT);
}

/* Hart h's spawn line (KMX_SPAWN_* words). */
static inline volatile uint64_t* spawn_line(uint64_t h) {
    return (volatile uint64_t*)(lim() + KMX_LIM_HARTS + 64u * h);
}

static inline const volatile kmx_peers* peers(void) {
    return (const volatile kmx_peers*)(lim() + KMX_LIM_PEERS);
}

/* A kmx_par_all line (KMX_CALL_* words) of this cluster's LIM, or through the window
 * onto cluster c's. */
static inline volatile uint32_t* all_line(volatile uint8_t* block, uint32_t line) {
    return (volatile uint32_t*)(block + peers()->all + KMX_ALL_CALL + 64u * line);
}

/* Records the first library error. */
static void fail(uint32_t error) {
    volatile kmx_result_block* r = result();
    if (r->error == KMX_ERR_NONE) {
        r->error = error;
        x280_fence_ow();
    }
}

/* KMX_OK, or why the run must end now: a stop was asked for, or it is late. */
static int must_end(uint64_t h) {
    x280_fence_ir();
    const volatile uint32_t* stop = (const volatile uint32_t*)(lim() + KMX_LIM_STOP);
    if (*stop == KMX_STOP_WORD ||
        x280_hart_stop_requested(h, x280_hart_control(h)->run_value)) {
        return KMX_STOPPED;
    }
    return x280_mtime() > g_deadline[h] ? KMX_LATE : KMX_OK;
}

/* Flush64 of every line of [addr, addr + bytes): a 4-byte store per line. */
static void flush(uint64_t addr, uint64_t bytes) {
    volatile uint64_t* reg = (volatile uint64_t*)(uintptr_t)KMX_CCACHE_FLUSH64;
    x280_fence_iorw();
    for (uint64_t a = addr & ~63ull; a < addr + bytes; a += 64) {
        __asm__ volatile(".option push\n.option norvc\nsd %0, 0(%1)\n.option pop\n"
                         :
                         : "r"(a), "r"(reg)
                         : "memory");
    }
    x280_fence_iorw();
}

const kmx_args* kmx_args_get(void) {
    return (const kmx_args*)lim();
}

int kmx_cluster(void) {
    return (int)kmx_args_get()->cluster;
}

void* kmx_region(int id, uint64_t* bytes) {
    const kmx_args* a = kmx_args_get();
    if (id < 0 || (uint32_t)id >= a->nregions) {
        if (bytes) {
            *bytes = 0;
        }
        return 0;
    }
    if (bytes) {
        *bytes = a->region[id].bytes;
    }
    return (void*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + a->region[id].offset);
}

/* A hand-over sweep split by lines over the four harts: hart h takes its quarter. */
typedef struct {
    uint64_t addr, lines;
} sweep_t;

static void sweep_part(int hart, void* arg) {
    const sweep_t* s = (const sweep_t*)arg;
    const uint64_t per = (s->lines + 3) / 4, first = (uint64_t)hart * per;
    if (first < s->lines) {
        const uint64_t n = s->lines - first < per ? s->lines - first : per;
        flush(s->addr + first * 64, n * 64);
    }
}

/* Sweeps [addr, addr + bytes): on hart 0 outside kmx_par over the four harts. */
static void sweep(uint64_t addr, uint64_t bytes) {
    if (hart_id() != 0 || g_in_par) {
        flush(addr, bytes); /* inside a kmx_par call: this hart alone */
        return;
    }
    const uint64_t lo = addr & ~63ull, hi = (addr + bytes + 63) & ~63ull;
    sweep_t s = {lo, (hi - lo) / 64};
    if (kmx_par(sweep_part, &s) != KMX_OK) {
        flush(addr, bytes); /* busy, late or stopped: the hand-over is still whole */
    }
}

void kmx_take_at(int id, uint64_t offset, uint64_t bytes) {
    uint64_t n;
    uint8_t* p = (uint8_t*)kmx_region(id, &n);
    if (p && offset < n) {
        sweep(
            (uint64_t)(uintptr_t)(p + offset), bytes < n - offset ? bytes : n - offset
        );
    }
}

void kmx_take(int id) {
    kmx_take_at(id, 0, ~0ull);
}

void kmx_give(int id) {
    kmx_take(id); /* Flush64 both writes back and drops a line */
}

void kmx_give_at(int id, uint64_t offset, uint64_t bytes) {
    kmx_take_at(id, offset, bytes);
}

/* KMX_OK on hart 0; elsewhere the error is recorded and the call refused. */
static int ring_hart(void) {
    if (hart_id() != 0) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    return KMX_OK;
}

int kmx_issue(const km_record_t* r, uint32_t* ticket) {
    const int h = ring_hart();
    return h != KMX_OK ? h : kmx_ring_issue(&g_ring, r, ticket);
}

int kmx_wait(uint32_t ticket, uint64_t deadline, kmx_event* ev) {
    int e = ring_hart();
    if (e == KMX_OK) {
        e = kmx_ring_wait(&g_ring, ticket, deadline, ev);
    }
    if (e == KMX_LATE || (e == KMX_STOPPED && g_ring.cmd != 0)) {
        fail(e == KMX_LATE ? KMX_ERR_LATE : KMX_ERR_STOPPED);
    }
    return e;
}

int kmx_poll(kmx_event* ev) {
    const int h = ring_hart();
    return h != KMX_OK ? h : kmx_ring_poll(&g_ring, ev);
}

/* Records the error of a wait that ended early and returns its code. */
static int ended(int end) {
    fail(end == KMX_LATE ? KMX_ERR_LATE : KMX_ERR_STOPPED);
    return end;
}

/* Hart 0: the spawns started on each hart (as its line's GEN word holds), and the
 * harts kmx_spawn started that no join has seen finish (bit h). */
static uint64_t g_spawned[4] __attribute__((section(".data"))) = {0, 0, 0, 0};
static uint32_t g_spawn_busy __attribute__((section(".data"))) = 0;

/* Harts [lo, hi): fn and arg into each line, then each line's next number. */
static void start(uint64_t lo, uint64_t hi, void (*fn)(int, void*), void* arg) {
    volatile uint64_t* line = spawn_line(0);
    for (uint64_t h = lo; h < hi; h++) {
        line[8 * h + KMX_SPAWN_FN] = (uint64_t)(uintptr_t)fn;
        line[8 * h + KMX_SPAWN_ARG] = (uint64_t)(uintptr_t)arg;
    }
    x280_fence_ow();
    for (uint64_t h = lo; h < hi; h++) {
        line[8 * h + KMX_SPAWN_GEN] = ++g_spawned[h];
    }
    x280_fence_ow();
}

/* Whether one of harts [lo, hi) has not finished its last spawn. */
static int busy(uint64_t lo, uint64_t hi) {
    x280_fence_ir();
    for (uint64_t h = lo; h < hi; h++) {
        if (spawn_line(h)[KMX_SPAWN_DONE] < g_spawned[h]) {
            return 1;
        }
    }
    return 0;
}

int kmx_spawn(int hart, void (*fn)(int hart, void* arg), void* arg) {
    if (hart_id() != 0) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    if (hart < 1 || hart > 3) {
        return KMX_BAD;
    }
    if (busy((uint64_t)hart, (uint64_t)hart + 1)) {
        return KMX_FULL;
    }
    start((uint64_t)hart, (uint64_t)hart + 1, fn, arg);
    g_spawn_busy |= 1u << hart;
    return KMX_OK;
}

/* Hart 0: waits until harts [lo, hi) finished their spawns. */
static int join(uint64_t lo, uint64_t hi) {
    const volatile uint64_t* line = spawn_line(0);
    const uint64_t g1 = g_spawned[1], g2 = g_spawned[2], g3 = g_spawned[3];
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if ((lo > 1 || line[8 + KMX_SPAWN_DONE] >= g1) &&
            (lo > 2 || hi < 3 || line[16 + KMX_SPAWN_DONE] >= g2) &&
            (hi < 4 || line[24 + KMX_SPAWN_DONE] >= g3)) {
            x280_fence_iorw();
            return KMX_OK;
        }
        if (spins % KMX_POLL_EVERY == 0) {
            const int end = must_end(0);
            if (end != KMX_OK) {
                return ended(end);
            }
        }
    }
}

int kmx_join(int hart) {
    if (hart_id() != 0) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    if (hart < 1 || hart > 3) {
        return KMX_BAD;
    }
    const int e = join((uint64_t)hart, (uint64_t)hart + 1);
    if (e == KMX_OK) {
        g_spawn_busy &= ~(1u << hart);
    }
    return e;
}

int kmx_par(void (*fn)(int hart, void* arg), void* arg) {
    if (hart_id() != 0 || g_in_par) {
        fail(KMX_ERR_PAR); /* another hart, or a kmx_par inside one */
        return KMX_STOPPED;
    }
    if (g_spawn_busy) {
        if (busy(1, 4)) {
            return KMX_FULL; /* a spawned hart has not finished */
        }
        g_spawn_busy = 0;
    }
    g_in_par = 1;
    start(1, 4, fn, arg);
    fn(0, arg);
    const int e = join(1, 4);
    g_in_par = 0;
    return e;
}

uint32_t kmx_clusters(void) {
    const volatile kmx_peers* p = peers();
    return p->lead ? p->mask : 1u << kmx_cluster();
}

int kmx_lead(void) {
    const uint32_t lead = peers()->lead;
    return lead ? (int)lead : kmx_cluster();
}

/* 32-bit stores of a 64-bit value through a window, low half first. */
static void put64(volatile uint32_t* w, uint64_t v) {
    w[0] = (uint32_t)v;
    w[1] = (uint32_t)(v >> 32);
}

static uint64_t get64(const volatile uint32_t* w) {
    return (uint64_t)w[0] | (uint64_t)w[1] << 32;
}

/* This cluster's done line in the lead's LIM: number `gen` finished (0: it serves). */
static void tell_lead(uint32_t gen) {
    volatile uint32_t* d = all_line(g_peer[peers()->lead], (uint32_t)kmx_cluster());
    if (gen == 0) {
        d[KMX_CALL_GEN] = 0;
        x280_fence_ow();
        d[KMX_CALL_NONCE] = peers()->nonce;
        x280_fence_ow();
    }
    d[KMX_CALL_GEN] = gen;
    x280_fence_ow();
}

/* Lead: waits until cluster c's done line holds this load's nonce and `gen`. */
static int wait_done(uint32_t c, uint32_t gen) {
    const volatile uint32_t* d = all_line(lim(), c);
    const uint32_t nonce = peers()->nonce;
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (d[KMX_CALL_NONCE] == nonce && d[KMX_CALL_GEN] >= gen) {
            x280_fence_iorw();
            return KMX_OK;
        }
        if (spins % KMX_POLL_EVERY == 0) {
            const int end = must_end(0);
            if (end != KMX_OK) {
                return ended(end);
            }
        }
    }
}

/* Lead: number `gen` of the call line into cluster c's LIM (exit: the last one). */
static void call(uint32_t c, uint32_t gen, uint64_t fn, uint64_t arg, uint32_t exit) {
    volatile uint32_t* l = all_line(g_peer[c], 0);
    put64(l + KMX_CALL_FN, fn);
    put64(l + KMX_CALL_ARG, arg);
    l[KMX_CALL_EXIT] = exit;
    l[KMX_CALL_NONCE] = peers()->nonce;
    x280_fence_ow();
    l[KMX_CALL_GEN] = gen;
    x280_fence_ow();
}

/* What every hart of a cluster runs for one kmx_par_all call. */
typedef struct {
    void (*fn)(int cluster, int hart, uint64_t arg);
    uint64_t arg;
    int cluster;
} all_call_t;

static void all_part(int hart, void* arg) {
    const all_call_t* c = (const all_call_t*)arg;
    c->fn(c->cluster, hart, c->arg);
}

/* The other clusters of the lead's group, as bits. */
static uint32_t others(void) {
    const volatile kmx_peers* p = peers();
    return p->lead ? p->mask & ~(1u << kmx_cluster()) : 0;
}

/* KMX_OK once every other cluster of the group serves: on the lead after each said
 * so, on another once the lead's first call came (KMX_BAD before). */
static int all_ready(void) {
    if (g_all_ready) {
        return KMX_OK;
    }
    if (kmx_lead() != kmx_cluster()) {
        return KMX_BAD;
    }
    const uint32_t rest = others();
    for (uint32_t c = 1; c < 4; c++) {
        const int e = (rest >> c & 1u) ? wait_done(c, 0) : KMX_OK;
        if (e != KMX_OK) {
            return e;
        }
    }
    g_all_ready = 1;
    return KMX_OK;
}

/* The lead's X280 core 0, not nested: KMX_OK, or KMX_STOPPED (library error) elsewhere.
 */
static int all_lead(void) {
    if (hart_id() != 0 || g_in_par || kmx_lead() != kmx_cluster()) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    return KMX_OK;
}

/* kmx_par_all with `part` (given an all_call_t) as the lead cluster's kmx_par part. */
static int all_run(
    void (*fn)(int, int, uint64_t), uint64_t arg, void (*part)(int, void*)
) {
    const uint32_t rest = others();
    int e = all_ready();
    if (e != KMX_OK) {
        return e;
    }
    const uint32_t gen = ++g_all_gen;
    for (uint32_t c = 1; c < 4; c++) {
        if (rest >> c & 1u) {
            call(c, gen, (uint64_t)(uintptr_t)fn, arg, 0);
        }
    }
    all_call_t mine = {fn, arg, kmx_cluster()};
    e = kmx_par(part, &mine);
    for (uint32_t c = 1; c < 4 && e == KMX_OK; c++) {
        e = (rest >> c & 1u) ? wait_done(c, gen) : KMX_OK;
    }
    return e;
}

int kmx_par_all(void (*fn)(int cluster, int hart, uint64_t arg), uint64_t arg) {
    const int e = all_lead();
    return e != KMX_OK ? e : all_run(fn, arg, all_part);
}

int kmx_serve(void) {
    if (hart_id() != 0 || g_in_par) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    if (kmx_lead() == kmx_cluster()) {
        return KMX_NONE;
    }
    const volatile uint32_t* l = all_line(lim(), 0);
    const uint32_t nonce = peers()->nonce;
    uint32_t seen = 0;
    tell_lead(0);
    for (uint32_t spins = 1;; spins++) {
        x280_fence_ir();
        const uint32_t gen = l[KMX_CALL_GEN];
        if (gen == seen + 1 && l[KMX_CALL_NONCE] == nonce) {
            x280_fence_ir();
            if (l[KMX_CALL_EXIT]) {
                return KMX_OK;
            }
            all_call_t mine = {
                (void (*)(int, int, uint64_t))(uintptr_t)get64(l + KMX_CALL_FN),
                get64(l + KMX_CALL_ARG),
                kmx_cluster()
            };
            seen = gen;
            g_all_ready = 1;
            const int e = kmx_par(all_part, &mine);
            if (e != KMX_OK) {
                return e;
            }
            tell_lead(gen);
            continue;
        }
        if (spins % KMX_POLL_EVERY == 0) {
            const int end = must_end(0);
            if (end != KMX_OK) {
                return ended(end);
            }
            if (seen == 0 && spins % (KMX_POLL_EVERY * 256u) == 0) {
                tell_lead(0); /* again: the lead clears its lines when it starts */
            }
        }
    }
}

/* Region `id` of cluster c's thread: its bank address and size, from the argument
 * block in its LIM (a read through the window); 0 for an unknown one. */
static uint64_t region_of(uint32_t c, int id, uint64_t* bytes) {
    *bytes = 0;
    const volatile uint8_t* block = c == (uint32_t)kmx_cluster() ? lim() : g_peer[c];
    if (block == 0 || id < 0) {
        return 0;
    }
    const volatile uint32_t* a = (const volatile uint32_t*)block;
    const uint32_t nregions = a[offsetof(kmx_args, nregions) / 4];
    if ((uint32_t)id >= nregions || nregions > KMX_REGIONS_MAX) {
        return 0;
    }
    const volatile uint32_t* r = a + (KMX_ARGS_REGIONS + 16u * (uint32_t)id) / 4;
    *bytes = get64(r + 2);
    return get64(r);
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

/* One hart's push of [src, src + n) to bank address `at` of DRAM tile `tile`: posted
 * stores per 2 MiB, the last line again, acknowledged. 0, or 1 if a window failed. */
static int push_range(
    uint64_t h, uint32_t tile, uint64_t at, const uint8_t* src, uint64_t n
) {
    const uint32_t x = tile & 0xFFFFu, y = tile >> 16;
    const uint32_t win = KMX_PUSH_WINDOW + (uint32_t)h;
    const uint64_t end = at + n;
    noc_tlb_2m_t prog;
    while (at < end) {
        const uint64_t room = NOC_2M_WINDOW_STRIDE - (at & (NOC_2M_WINDOW_STRIDE - 1u));
        const uint64_t k = end - at < room ? end - at : room;
        volatile uint8_t* w = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                win, x, y, at, NOC_ORDER_POSTED, &prog
            )
        );
        if (w == 0) {
            return 1;
        }
        copy_out(w, src, k);
        x280_fence_ow();
        at += k;
        src += k;
    }
    /* The last line again through a non-posted window: its acknowledgement comes
     * after the posted stores before it on the same path. */
    volatile uint32_t* d = (
        (volatile uint32_t*)noc_configure_tlb_2m_checked(
            win, x, y, end - 64, NOC_ORDER_DEFAULT, &prog
        )
    );
    if (d == 0) {
        return 1;
    }
    const volatile uint32_t* s = (const volatile uint32_t*)(src - 64);
    for (int i = 0; i < 16; i++) {
        d[i] = s[i];
    }
    x280_fence_ow();
    noc_clear_tlb_2m(win);
    return 0;
}

typedef struct {
    uint32_t tile;
    uint64_t at, bytes;
    const uint8_t* src;
    uint32_t bad[4];
} push_t;

/* Hart h's quarter, in whole lines. */
static void push_part(int hart, void* arg) {
    push_t* p = (push_t*)arg;
    const uint64_t lines = p->bytes / 64, per = (lines + 3) / 4;
    const uint64_t first = (uint64_t)hart * per;
    p->bad[hart] = 0;
    if (first < lines) {
        const uint64_t n = (lines - first < per ? lines - first : per) * 64;
        p->bad[hart] = (
            (uint32_t)push_range(
                (uint64_t)hart, p->tile, p->at + first * 64, p->src + first * 64, n
            )
        );
    }
}

int kmx_push(int cluster, int id, uint64_t offset, const void* src, uint64_t bytes) {
    const uint32_t c = (uint32_t)cluster;
    if (cluster < 1 || cluster > 3 || !(kmx_clusters() >> c & 1u) ||
        ((offset | bytes | (uint64_t)(uintptr_t)src) & 63u)) {
        return KMX_BAD;
    }
    if (c != (uint32_t)kmx_cluster()) {
        const int e = all_ready(); /* its argument block is this load's */
        if (e != KMX_OK) {
            return e;
        }
    }
    uint64_t n;
    const uint64_t at = region_of(c, id, &n);
    if (n == 0 || offset > n || bytes > n - offset) {
        return KMX_BAD;
    }
    if (bytes == 0) {
        return KMX_OK;
    }
    push_t p = {
        peers()->bank[c], at + offset, bytes, (const uint8_t*)src, {0, 0, 0, 0}
    };
    const uint64_t h = hart_id();
    if (h == 0 && !g_in_par && kmx_par(push_part, &p) == KMX_OK) {
        return p.bad[0] | p.bad[1] | p.bad[2] | p.bad[3] ? KMX_BAD : KMX_OK;
    }
    return push_range(h, p.tile, p.at, p.src, bytes) ? KMX_BAD : KMX_OK;
}

/* X280 ports, per core: KM_CTL_X280 through its window, serving X280 core + 1, last ask
 * answered, tell seen, tell sent; few words read (LIM data costs ~30 cycles a load). */
#define KMX_L1_BYTES 0x180000u /* a Tensix core's L1 */
KMX_STATIC_ASSERT(KMX_PORTS_CORES == KM_CORES_MAX, "ports: one entry per core");
static volatile uint8_t* g_port[KMX_PORTS_CORES]
    __attribute__((section(".data"))) = {0};
static uint32_t g_port_who[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static const volatile uint8_t* g_box __attribute__((section(".data"))) = 0;
static uint32_t g_answered[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static uint32_t g_told[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static uint32_t g_tells[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static uint32_t g_port_cores __attribute__((section(".data"))) = 0;

static inline const volatile kmx_ports* ports(void) {
    return (const volatile kmx_ports*)(lim() + KMX_LIM_PORTS);
}

/* Opens a window per core this cluster serves (the area was written before any
 * start): 1, or 0 for a line out of range or a window that did not read back. */
static int ports_open(void) {
    const volatile kmx_ports* p = ports();
    const uint32_t at = p->at, ctl = p->ctl, cores = p->cores;
    if (at == 0) {
        return 1;
    }
    if ((at & 63u) || at < x280_hart_args_bytes() ||
        (uint64_t)(uintptr_t)lim() + at + KMX_PORTS_BYTES > X280_LIM_END ||
        cores == 0 || cores > KMX_PORTS_CORES || (ctl & 63u) ||
        ctl + KM_CTL_BYTES > KMX_L1_BYTES || (p->flags & ~KMX_PORTS_NOC1)) {
        return 0;
    }
    const volatile uint8_t* area = lim() + at;
    const uint32_t me = (uint32_t)kmx_cluster();
    for (uint32_t i = 0; i < cores; i++) {
        const uint32_t owner = area[KMX_PORTS_OWNER + i];
        if (owner >> 2 != me) {
            continue;
        }
        const uint32_t xy = ((const volatile uint32_t*)(area + KMX_PORTS_TILES))[i];
        const uint32_t x = xy & 0xFFFFu, y = xy >> 16;
        if (x > NOC_MAX_COORD || y > NOC_MAX_COORD) {
            return 0;
        }
        noc_tlb_2m_t prog;
        prog.data[0] = prog.data[1] = prog.data[2] = prog.data[3] = 0;
        noc_fill_tlb_2m(&prog, x, y, x, y, 0, NOC_ORDER_POSTED, 0, 0u);
        prog.noc_sel = (p->flags & KMX_PORTS_NOC1) ? 1u : 0u;
        volatile uint8_t* w = (
            (volatile uint8_t*)noc_configure_tlb_2m_ext(KMX_PORT_WINDOW + i, &prog, 0)
        );
        g_port[i] = w + ctl + KM_CTL_X280;
        g_port_who[i] = (owner & 3u) + 1u;
        if (noc_verify_tlb_2m(KMX_PORT_WINDOW + i, &prog) != 0) {
            return 0;
        }
    }
    g_box = area + KMX_PORTS_BOX;
    g_port_cores = cores;
    x280_fence_ow();
    return 1;
}

/* Every port window this cluster opened is cleared. */
static void ports_close(void) {
    x280_fence_ow();
    for (uint32_t i = 0; i < KMX_PORTS_CORES; i++) {
        if (g_port[i] != 0) {
            noc_clear_tlb_2m(KMX_PORT_WINDOW + i);
            g_port[i] = 0;
        }
        g_port_who[i] = 0;
    }
    g_port_cores = 0;
}

/* Whether the calling X280 core serves `core`. */
static inline int port_mine(int core) {
    return (uint32_t)core < KMX_PORTS_CORES && g_port_who[core] == hart_id() + 1u;
}

/* One 16-byte store of {seq, a, b, c} into line `at` of the core's KM_CTL_X280. */
static void port_store(int core, uint32_t at, const uint32_t* line) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(line), "r"(g_port[core] + at)
        : "memory", "v8"
    );
    x280_fence_ow();
}

/* Line `at` of the core's mailbox as one 16-byte load, and its seq again: the two
 * must agree (the core's 16 bytes land whole). Torn on every try: no message. */
static void box_read(int core, uint32_t at, kmx_msg* m) {
    const volatile uint32_t* b = (const volatile uint32_t*)(g_box + 64u * core + at);
    for (int i = 0; i < 4; i++) {
        x280_fence_ir();
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 4, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"(b), "r"(m)
            : "memory", "v8"
        );
        if (b[0] == m->seq) {
            return;
        }
    }
    m->seq = 0;
}

int kmx_port_cores(void) {
    return (int)g_port_cores;
}

int kmx_port_owner(int core) {
    return (uint32_t)core < KMX_PORTS_CORES ? (int)g_port_who[core] - 1 : -1;
}

int kmx_port_tell(int core, uint32_t a, uint32_t b, uint32_t c) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    const uint32_t line[4] = {++g_tells[core], a, b, c};
    port_store(core, KM_X280_TELL, line);
    return KMX_OK;
}

int kmx_port_asked(int core, kmx_msg* m) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    box_read(core, KM_BOX_ASK, m);
    return m->seq != 0 && m->seq != g_answered[core] ? KMX_OK : KMX_NONE;
}

int kmx_port_answer(int core, uint32_t seq, uint32_t a, uint32_t b, uint32_t c) {
    if (!port_mine(core) || seq == 0) {
        return KMX_BAD;
    }
    const uint32_t line[4] = {seq, a, b, c};
    port_store(core, KM_X280_ANSWER, line);
    g_answered[core] = seq;
    return KMX_OK;
}

int kmx_port_told(int core, kmx_msg* m) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    box_read(core, KM_BOX_TELL, m);
    if (m->seq == 0 || m->seq == g_told[core]) {
        return KMX_NONE;
    }
    g_told[core] = m->seq;
    return KMX_OK;
}

/* kmx_port_run on the lead's X280 core 0: inside one, its ticket, the run's number,
 * whether it ended and why (KMX_OK: its event, in g_port_ev). */
static uint32_t g_port_live __attribute__((section(".data"))) = 0;
static uint32_t g_port_ticket __attribute__((section(".data"))) = 0;
static uint32_t g_port_n __attribute__((section(".data"))) = 0;
static uint32_t g_port_ended __attribute__((section(".data"))) = 0;
static int g_port_why __attribute__((section(".data"))) = KMX_OK;
static kmx_event g_port_ev __attribute__((section(".data"))) = {0, 0, 0, 0};
static uint32_t g_open_calls[4] __attribute__((section(".data"))) = {0, 0, 0, 0};

/* The run state into the ports line of every cluster of the group. */
static void port_state(uint32_t v) {
    const uint32_t at = KMX_LIM_PORTS + offsetof(kmx_ports, state);
    *(volatile uint32_t*)(lim() + at) = v;
    const uint32_t rest = others();
    for (uint32_t c = 1; c < 4; c++) {
        if ((rest >> c & 1u) && g_peer[c] != 0) {
            *(volatile uint32_t*)(g_peer[c] + at) = v;
        }
    }
    x280_fence_ow();
}

/* The lead's X280 core 0: the run ends, for `why`, on every X280 core of the group. */
static void port_end(int why) {
    g_port_why = why;
    g_port_ended = 1;
    port_state(g_port_n << 1 | 1u);
}

int kmx_port_open(void) {
    const uint64_t h = hart_id();
    const uint32_t n = ++g_open_calls[h], check = n % KMX_PORT_CHECK == 0;
    if (h == 0 && g_port_live) {
        if (g_port_ended) {
            return 0;
        }
        if (n % KMX_PORT_LOOK != 0) {
            return 1;
        }
        const int e = kmx_ring_look(&g_ring, g_port_ticket, &g_port_ev);
        if (e != KMX_NONE) {
            port_end(e);
            return 0;
        }
        const int end = check ? must_end(0) : KMX_OK;
        if (end != KMX_OK) {
            port_end(end);
            return 0;
        }
        return 1;
    }
    x280_fence_ir();
    const uint32_t state = ports()->state;
    if (state == 0 || (state & 1u) || (check && must_end(h) != KMX_OK)) {
        return 0;
    }
    return 1;
}

/* The lead cluster's part of a kmx_port_run: fn, then on X280 core 0 the end. */
static void port_part(int cpu, void* arg) {
    all_part(cpu, arg);
    while (cpu == 0 && kmx_port_open()) {
        ;
    }
}

int kmx_port_run(
    const km_record_t* r,
    void (*fn)(int cluster, int cpu, uint64_t arg),
    uint64_t arg,
    kmx_event* ev
) {
    int e = all_lead();
    if (e != KMX_OK) {
        return e;
    }
    if (g_port_cores == 0) {
        return KMX_BAD;
    }
    e = all_ready();
    uint32_t ticket = 0;
    e = e == KMX_OK ? kmx_issue(r, &ticket) : e;
    if (e != KMX_OK) {
        return e;
    }
    g_port_ticket = ticket;
    g_port_ended = 0;
    g_port_why = KMX_OK;
    g_port_live = 1;
    port_state(++g_port_n << 1);
    e = all_run(fn, arg, port_part);
    g_port_live = 0;
    if (!g_port_ended) {
        port_end(e);
    }
    if (e != KMX_OK) {
        return e;
    }
    if (g_port_why == KMX_LATE || g_port_why == KMX_STOPPED) {
        return ended(g_port_why);
    }
    if (g_port_why == KMX_OK) {
        *ev = g_port_ev;
    }
    return g_port_why;
}

uint64_t kmx_time(void) {
    return x280_mtime();
}

int kmx_stop_requested(void) {
    return must_end(hart_id()) != KMX_OK;
}

void kmx_progress(uint32_t where) {
    volatile kmx_result_block* r = result();
    r->where = where;
    r->t_where = (uint32_t)(x280_mtime() - g_start);
    x280_fence_ow();
}

void kmx_result(uint32_t code, const void* words, uint32_t n) {
    volatile kmx_result_block* r = result();
    if (n > KMX_RESULT_WORDS_MAX) {
        fail(KMX_ERR_RESULT);
        n = KMX_RESULT_WORDS_MAX;
    }
    const uint32_t* w = (const uint32_t*)words;
    for (uint32_t i = 0; i < n; i++) {
        r->words[i] = w[i];
    }
    r->n = n;
    r->code = code;
    g_code = code;
    x280_fence_ow();
}

/* Harts 1 to 3: run each spawn of hart 0 until kmx_main returns, a stop is asked
 * for, or the run's deadline passes. */
static void serve(uint64_t h) {
    volatile uint64_t* line = spawn_line(h);
    const volatile uint64_t* exit = spawn_line(0);
    uint64_t seen = 0;
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (exit[KMX_SPAWN_EXIT] != 0) {
            return;
        }
        const uint64_t gen = line[KMX_SPAWN_GEN];
        if (gen > seen) {
            x280_fence_ir();
            void (*fn)(
                int, void*
            ) = (void (*)(int, void*))(uintptr_t)line[KMX_SPAWN_FN];
            fn((int)h, (void*)(uintptr_t)line[KMX_SPAWN_ARG]);
            x280_fence_iorw();
            line[KMX_SPAWN_DONE] = gen;
            x280_fence_ow();
            seen = gen;
            continue;
        }
        if (spins % KMX_POLL_EVERY == 0 && must_end(h) != KMX_OK) {
            return;
        }
    }
}

/* Opens a window onto the LIM of every other cluster of the group and, on the lead,
 * clears the done lines. 1, or 0 when a window could not be opened. */
static int peers_open(void) {
    const volatile kmx_peers* p = peers();
    const uint32_t rest = others();
    if (p->lead == 0) {
        return 1;
    }
    if (p->lead > 3 || !(p->mask >> p->lead & 1u) || (p->mask & ~0xEu) ||
        !(p->mask >> kmx_cluster() & 1u) || (p->all & 63u) || p->all < KMX_LIM_BYTES ||
        (uint64_t)(uintptr_t)lim() + p->all + KMX_ALL_BYTES > X280_LIM_END) {
        return 0;
    }
    for (uint32_t c = 1; c < 4; c++) {
        if (!(rest >> c & 1u)) {
            continue;
        }
        noc_tlb_2m_t prog;
        volatile uint8_t* w = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                KMX_PEER_WINDOW + c - 1,
                p->tile[c] & 0xFFFFu,
                p->tile[c] >> 16,
                X280_LIM_BASE,
                NOC_ORDER_POSTED,
                &prog
            )
        );
        if (w == 0) {
            return 0;
        }
        /* every cluster's block is at the same LIM address */
        g_peer[c] = w + ((uint64_t)(uintptr_t)lim() - X280_LIM_BASE);
    }
    if (kmx_lead() == kmx_cluster()) {
        for (uint32_t c = 1; c < 4; c++) {
            volatile uint32_t* d = all_line(lim(), c);
            d[KMX_CALL_GEN] = 0;
            d[KMX_CALL_NONCE] = 0;
        }
        x280_fence_ow();
    }
    return 1;
}

/* The lead tells the others its kmx_main returned; every window is closed. */
static void peers_close(void) {
    const uint32_t rest = others();
    for (uint32_t c = 1; c < 4; c++) {
        if (g_peer[c] != 0 && kmx_lead() == kmx_cluster()) {
            call(c, g_all_gen + 1, 0, 0, 1);
        }
        if (rest >> c & 1u) {
            noc_clear_tlb_2m(KMX_PEER_WINDOW + c - 1);
            g_peer[c] = 0;
        }
    }
}

/* Whether the argument block is this run's and every entry is in range. */
static int args_valid(const volatile kmx_args* a, uint64_t run) {
    if (x280_hart_args_bytes() < KMX_ARGS_BYTES || a->run_value != run ||
        a->magic != KMX_MAGIC || a->version != KMX_VERSION || a->deadline == 0 ||
        a->deadline > KMX_DEADLINE_MAX || a->nregions > KMX_REGIONS_MAX ||
        a->cluster < 1 || a->cluster > 3 || a->nwords > KMX_WORDS_MAX ||
        (a->check & 63u)) {
        return 0;
    }
    for (uint32_t i = 0; i < a->nregions; i++) {
        const uint64_t at = a->region[i].offset, n = a->region[i].bytes;
        if ((at & 63u) || n == 0 || at + n < at || at + n > (1ull << 32) ||
            (at <= X280_IPI_ACK_OFFSET && X280_IPI_ACK_OFFSET < at + n)) {
            return 0;
        }
    }
    return 1;
}

/* Opens the ring the line names: as it is, or for KMX_RING_REMOTE through a window
 * onto the commands and the read line. 1, or 0 when the line is out of range. */
static int ring_open(const volatile kmx_ring_desc* d) {
    kmx_ring_desc l;
    l.cmd = d->cmd;
    l.consumed = d->consumed;
    l.events = d->events;
    l.read = d->read;
    l.cmd_depth = d->cmd_depth;
    l.event_depth = d->event_depth;
    l.flags = d->flags;
    if (l.flags & KMX_RING_REMOTE) {
        const uint64_t base = l.cmd & ~(NOC_2M_WINDOW_STRIDE - 1u);
        const uint32_t xy = d->reserved[0], x = xy & 0xFFFFu, y = xy >> 16;
        if ((l.read & ~(NOC_2M_WINDOW_STRIDE - 1u)) != base || (l.flags & 1u) ||
            l.read + 64u > base + NOC_2M_WINDOW_STRIDE) {
            return 0;
        }
        noc_tlb_2m_t programmed;
        volatile uint8_t* win = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                KMX_RING_WINDOW, x, y, base, NOC_ORDER_POSTED, &programmed
            )
        );
        if (win == 0 || noc_verify_tlb_2m(KMX_RING_WINDOW, &programmed) != 0) {
            return 0;
        }
        l.cmd = (uint64_t)(uintptr_t)win + (l.cmd - base);
        l.read = (uint64_t)(uintptr_t)win + (l.read - base);
    }
    if (!kmx_ring_open(&g_ring, &l)) {
        if (l.flags & KMX_RING_REMOTE) {
            noc_clear_tlb_2m(KMX_RING_WINDOW);
        }
        return 0;
    }
    return 1;
}

/* Hart 0: the bank check, kmx_main, the result block. */
static uint64_t run_main(const volatile kmx_args* a) {
    volatile kmx_result_block* r = result();
    r->state = KMX_STATE_BOOT;
    x280_fence_ow();
    const volatile kmx_ring_desc* d = (
        (const volatile kmx_ring_desc*)(lim() + KMX_LIM_RING)
    );
    if (d->cmd != 0 && !ring_open(d)) {
        fail(KMX_ERR_ARGS); /* a ring is named but its line is out of range */
    }
    if (!peers_open()) {
        fail(KMX_ERR_ARGS); /* the peers line is out of range */
    }
    if (!ports_open()) {
        fail(KMX_ERR_ARGS); /* the ports line is out of range, or a window failed */
    }
    const uint64_t check = X280_LOCAL_DDR_TILE_BASE + a->check;
    flush(check, 64);
    if (*(volatile uint64_t*)(uintptr_t)check != a->check_value) {
        fail(KMX_ERR_BANK);
    }
    if (r->error == KMX_ERR_NONE) {
        r->state = KMX_STATE_RUNNING;
        x280_fence_ow();
        const int code = kmx_main(0);
        if (g_code == KMX_UNSET) {
            r->code = (uint32_t)code;
        }
    }
    if (g_ring.cmd != 0 && (d->flags & KMX_RING_REMOTE)) {
        noc_clear_tlb_2m(KMX_RING_WINDOW);
    }
    peers_close();
    ports_close();
    const uint64_t ticks = x280_mtime() - g_start;
    r->ticks = (uint32_t)ticks;
    /* every region given back: a dirty line left in the cluster's cache would be
     * written over a later load's data (or its check line) when it is evicted */
    for (uint32_t i = 0; i < a->nregions; i++) {
        uint64_t n;
        const void* p = kmx_region((int)i, &n);
        flush((uint64_t)(uintptr_t)p, n); /* hart 0 alone: as fast, and after a stop */
    }
    spawn_line(0)[KMX_SPAWN_EXIT] = 1;
    x280_fence_ow();
    r->state = r->error ? KMX_STATE_FAILED : KMX_STATE_DONE;
    x280_fence_ow();
    return r->error ? ~(uint64_t)r->error : ticks;
}

static __attribute__((noinline)) void boot(uint64_t h) {
    const uint64_t t0 = x280_mtime();
    const volatile kmx_args* a = (const volatile kmx_args*)lim();
    if (a == 0) {
        return;
    }
    const uint64_t run = x280_hart_control(h)->run_value;
    if (!args_valid(a, run)) {
        if (h == 0) {
            result()->error = KMX_ERR_ARGS;
            result()->state = KMX_STATE_FAILED;
            x280_fence_ow();
        }
        x280_hart_set_result(h, ~(uint64_t)KMX_ERR_ARGS);
        return;
    }
    g_deadline[h] = t0 + a->deadline;
    if (h != 0) {
        serve(h);
        return;
    }
    g_start = t0;
    x280_hart_set_result(0, run_main(a));
}

int main(uint64_t hartid) {
    /* Floating point and vector on first, on every hart: the harness leaves them off
     * and the first such instruction would trap. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    const volatile kmx_args* a = (const volatile kmx_args*)lim();
    if (a != 0 && !(a->flags & KMX_FLAG_KEEP_PREFETCH)) {
        volatile uint32_t* pf = (
            (volatile uint32_t*)(uintptr_t)(KMX_L2PF_BASE +
            (hartid & 3u) * KMX_L2PF_STRIDE)
        );
        pf[0] = KMX_L2PF_BASIC;
        pf[1] = KMX_L2PF_USER;
        x280_fence_iorw();
    }
    boot(hartid);
    return 0;
}
