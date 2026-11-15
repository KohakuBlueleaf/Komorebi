/* X280 ports experiment, the thread: commands to the grid, and every X280 core of every
 * cluster serving its cores' ports meanwhile (README.md, xp.h). */
#include <stdint.h>

#include "xp.h"
#include "kmx.h"
#include "noc.h"

#define WAIT_TICKS KMX_TICKS_PER_SECOND /* per command, longer than its deadline */
#define BLOCK_WINDOW 160u /* X280 core h's posted window onto a core: + h */
#define BLOCK_LAST 164u   /* X280 core h's non-posted one, for the block's last line */
#define SPARE_WINDOW 170u /* the window step (d) aims again and again */
#define FREE_FIRST 24u    /* windows an X280 core may hold: [FREE_FIRST, FREE_END) */
#define FREE_END 192u

/* Each X280 core's counts in this cluster, a line each. */
typedef struct {
    uint32_t answered, told, tells, done, ticks, passes, xfer, bad;
    uint32_t pad[8];
} hstat_t;

static hstat_t g_hs[4] __attribute__((section(".data"))) = {{0}};

static uint64_t cycles(void) {
    uint64_t c;
    __asm__ volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

/* Fills r field by field (a struct assignment may call memset, which does not link). */
static void command(
    km_command_t* r,
    uint32_t handler,
    uint32_t first,
    uint32_t count,
    uint32_t flags,
    const uint32_t* a
) {
    r->sequence = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = handler | flags << 8;
    r->group = first | count << 16;
    r->a0 = a[0];
    r->a1 = a[1];
    r->a2 = a[2];
    r->a3 = a[3];
    r->a4 = a[4];
    r->a5 = a[5];
    r->a6 = a[6];
    r->a7 = a[7];
    r->a8 = r->a9 = 0;
}

/* Submits r and waits for its event: KMX_OK and event, or the error. */
static int submit(const km_command_t* r, uint32_t* handle) {
    const uint64_t until = kmx_time() + WAIT_TICKS;
    int e = kmx_submit(r, handle);
    while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        e = kmx_submit(r, handle);
    }
    return e;
}

/* The tile of core i, from the ports area of the block. */
static uint32_t tile_of(int core) {
    const uint8_t* lim = (const uint8_t*)kmx_args_get();
    const kmx_ports* p = (const kmx_ports*)(lim + KMX_LIM_PORTS);
    return ((const uint32_t*)(lim + p->at + KMX_PORTS_TILES))[core];
}

static uint32_t control_of(void) {
    const uint8_t* lim = (const uint8_t*)kmx_args_get();
    return ((const kmx_ports*)(lim + KMX_LIM_PORTS))->control;
}

/* 64-byte vector copy into a window (n a multiple of 64). */
static void copy_lines(volatile uint8_t* dst, const uint32_t* src, uint32_t n) {
    for (uint32_t i = 0; i < n; i += 64) {
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 16, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"((const uint8_t*)src + i), "r"(dst + i)
            : "memory", "v8"
        );
    }
}

/* Round k's block for core c, made in region 0 at 4 KiB * `slot` (cached memory). */
static uint32_t* block_make(int core, uint32_t k, uint32_t slot) {
    uint64_t n;
    uint32_t* p = (uint32_t*)kmx_region(0, &n);
    if (p == 0 || (uint64_t)(slot + 1) * XP_BLOCK_BYTES > n) {
        return 0;
    }
    p += slot * (XP_BLOCK_BYTES / 4);
    for (uint32_t i = 0; i < XP_BLOCK_BYTES / 4; i++) {
        p[i] = XP_BLOCK_WORD(k, (uint32_t)core, i);
    }
    return p;
}

/* Round k's block into core c's L1 at `addr`: posted lines, then the last one again
 * non-posted (acknowledged after them); the copy's cycles into *xfer. */
static int block_l1(int cpu, int core, uint32_t addr, uint32_t k, uint32_t* xfer) {
    const uint32_t* b = block_make(core, k, KMX_PORTS_CORES + (uint32_t)cpu);
    if (b == 0 || addr + XP_BLOCK_BYTES > 0x180000u) {
        return 1;
    }
    const uint64_t t0 = cycles();
    const uint32_t xy = tile_of(core), x = xy & 0xFFFFu, y = xy >> 16;
    noc_tlb_2m_t prog;
    volatile uint8_t* w = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            BLOCK_WINDOW + cpu, x, y, 0, NOC_ORDER_POSTED, &prog
        )
    );
    volatile uint8_t* l = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            BLOCK_LAST + cpu, x, y, 0, NOC_ORDER_DEFAULT, &prog
        )
    );
    if (w == 0 || l == 0) {
        return 1;
    }
    copy_lines(w + addr, b, XP_BLOCK_BYTES);
    x280_fence_ow();
    volatile uint32_t* last = (volatile uint32_t*)(l + addr + XP_BLOCK_BYTES - 64);
    for (int i = 0; i < 16; i++) {
        last[i] = b[XP_BLOCK_BYTES / 4 - 16 + i];
    }
    x280_fence_ow();
    *xfer += (uint32_t)(cycles() - t0);
    return 0;
}

/* Round k's block for core c into region 0 at 4 KiB * c, flushed to the grid (the
 * flush's cycles go to *xfer). */
static int block_dram(int core, uint32_t k, uint32_t* xfer) {
    if (block_make(core, k, (uint32_t)core) == 0) {
        return 1;
    }
    const uint64_t t0 = cycles();
    kmx_flush_at(0, (uint64_t)core * XP_BLOCK_BYTES, XP_BLOCK_BYTES);
    *xfer += (uint32_t)(cycles() - t0);
    return 0;
}

/* One X280 core's serve in round `round`: answers (none after `stop` of them, 0: no
 * limit), `tell` a tell per core and pass, until each core told XP_DONE or the end. */
static void serve(int cluster, int cpu, uint64_t arg) {
    const uint32_t round = (uint32_t)arg & 0xFFFFFFu, tell = (uint32_t)(arg >> 24) & 1u;
    const uint32_t stop = (uint32_t)(arg >> 32);
    hstat_t* s = &g_hs[cpu];
    uint8_t mine[KMX_PORTS_CORES];
    int n = 0;
    for (int c = 0; c < kmx_port_cores(); c++) {
        if (kmx_port_owner(c) == cpu) {
            mine[n++] = (uint8_t)c;
        }
    }
    uint32_t done[KMX_PORTS_CORES / 32] = {0, 0, 0, 0};
    int left = n;
    const uint64_t t0 = kmx_time();
    uint32_t answered = 0, told = 0, tells = 0, passes = 0, bad = 0, xfer = 0;
    while (left > 0 && kmx_port_open()) {
        passes++;
        for (int i = 0; i < n; i++) {
            const int c = mine[i];
            if (done[c / 32] >> (c % 32) & 1u) {
                continue;
            }
            kmx_message m;
            if (tell) {
                kmx_port_tell(c, tells++, round, 0);
            }
            if (kmx_port_asked(c, &m) == KMX_OK && (stop == 0 || answered < stop)) {
                int e = 0;
                if (m.w[0] == XP_ASK_BLOCK) {
                    e = block_l1(cpu, c, m.w[1], m.w[2], &xfer);
                } else if (m.w[0] == XP_ASK_DBLOCK) {
                    e = block_dram(c, m.w[2], &xfer);
                }
                bad += (uint32_t)e;
                if (m.w[0] == XP_ASK_DBLOCK) { /* where: this cluster's bank */
                    const kmx_args* a = kmx_args_get();
                    const uint32_t at = (
                        (uint32_t)a->region[0].offset + c * XP_BLOCK_BYTES
                    );
                    const uint32_t tile = a->words[cluster == 1 ? 5 : 6];
                    kmx_port_answer(c, m.sequence, at, tile, 0);
                } else {
                    kmx_port_answer(c, m.sequence, XP_ANSWER(m.w[1], m.w[2]), 0, 0);
                }
                answered++;
            }
            if (kmx_port_told(c, &m) == KMX_OK) {
                if (m.w[0] == XP_DONE && m.w[1] == round) {
                    done[c / 32] |= 1u << (c % 32);
                    left--;
                } else {
                    told++;
                }
            }
        }
    }
    s->answered += answered;
    s->told += told;
    s->tells += tells;
    s->done += (uint32_t)(n - left);
    s->passes += passes;
    s->bad += bad;
    s->xfer += xfer;
    s->ticks += (uint32_t)(kmx_time() - t0);
}

static void nothing(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)cpu;
    (void)arg;
}

/* This cluster's per-X280 core counts as result words after `head` words. */
static void finish(uint32_t code, uint32_t* w, uint32_t head) {
    w[head + 28] = (uint32_t)kmx_port_cores();
    for (int c = 0; c < kmx_port_cores(); c++) {
        w[head + 29] += kmx_port_owner(c) >= 0;
    }
    w[head + 30] = g_hs[0].bad + g_hs[1].bad + g_hs[2].bad + g_hs[3].bad;
    for (int h = 0; h < 4; h++) {
        w[head + 7 * h] = g_hs[h].answered;
        w[head + 7 * h + 1] = g_hs[h].told;
        w[head + 7 * h + 2] = g_hs[h].tells;
        w[head + 7 * h + 3] = g_hs[h].done;
        w[head + 7 * h + 4] = g_hs[h].ticks;
        w[head + 7 * h + 5] = g_hs[h].passes;
        w[head + 7 * h + 6] = g_hs[h].xfer;
    }
    for (int h = 0; h < 4; h++) {
        noc_clear_tlb_2m(BLOCK_WINDOW + h);
        noc_clear_tlb_2m(BLOCK_LAST + h);
    }
    kmx_result(code, w, head + 31);
}

/* Steps (a) and (b): one core, a command per round. */
static uint32_t one_core(const uint32_t* a, uint32_t* w) {
    const uint32_t mode = a[0], rounds = a[1], core = a[2];
    uint32_t bad = 0, first_bad = 0;
    for (uint32_t k = 1; k <= rounds; k++) {
        km_command_t r;
        const uint32_t value = XP_ANSWER(k, core);
        const uint32_t args[8] = {k * 1000u, k, 0, 0, 0, 0, 0, k};
        const uint32_t handler = (
            mode == XP_MODE_TELL ? XP_HANDLER_TELL : XP_HANDLER_SEND
        );
        command(&r, handler, core, 1, mode == XP_MODE_TELL ? KM_FLAG_RESULT : 0, args);
        uint32_t handle;
        if (submit(&r, &handle) != KMX_OK) {
            return 3;
        }
        int ok = 1;
        if (mode == XP_MODE_TELL) {
            kmx_port_tell((int)core, value, k, 0);
        } else {
            kmx_message m;
            int e = KMX_NONE;
            const uint64_t until = kmx_time() + WAIT_TICKS;
            while (
                (e = kmx_port_told((int)core, &m)) == KMX_NONE && kmx_time() < until &&
                !kmx_stop_requested()
            ) {
            }
            ok = e == KMX_OK && m.w[0] == k * 1000u + core && m.w[1] == k;
        }
        kmx_event event;
        if (kmx_wait(handle, kmx_time() + WAIT_TICKS, &event) != KMX_OK) {
            return 3;
        }
        if (event.kind != KM_EVENT_DONE) {
            return 4;
        }
        if (mode == XP_MODE_TELL) {
            ok = event.ticks == value;
        }
        if (!ok) {
            bad++;
            first_bad = first_bad ? first_bad : k;
        }
    }
    w[0] = rounds;
    w[1] = bad;
    w[2] = first_bad;
    return 0;
}

/* Steps (c), (e), (f), (g): a command to cores [0, cores) per round, served by every
 * X280 core of every cluster. */
static uint32_t all_cores(const uint32_t* a, uint32_t* w) {
    const uint32_t mode = a[0], rounds = a[1], cores = a[2];
    uint32_t kinds[4] = {0, 0, 0, 0};
    uint64_t ticks = 0;
    if (kmx_parallel_all(nothing, 0) !=
        KMX_OK) { /* every cluster serves before round 1 */
        return 3;
    }
    for (uint32_t k = 1; k <= rounds; k++) {
        km_command_t r;
        uint32_t handler = XP_HANDLER_ASK, tell = 0, stop = 0;
        uint32_t args[8] = {a[3], a[5], a[4] | XP_F_DONE, a[6], a[7], 0, 0, k};
        if (mode == XP_MODE_BLOCK) {
            handler = XP_HANDLER_BLOCK;
            args[0] = a[3];
            args[1] = a[4];
            args[2] = a[7];

        } else if (mode == XP_MODE_RATE) {
            handler = XP_HANDLER_RATE;
            args[0] = a[3];
            args[1] = a[4];
            tell = 1;
        } else {
            stop = a[8];
        }
        command(&r, handler, 0, cores, 0, args);
        const uint64_t t0 = kmx_time(), until = t0 + WAIT_TICKS;
        const uint64_t arg = k | (uint64_t)tell << 24 | (uint64_t)stop << 32;
        kmx_event event;
        int e = kmx_port_run(&r, serve, arg, &event);
        while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
            e = kmx_port_run(&r, serve, arg, &event);
        }
        if (e != KMX_OK) {
            return 3;
        }
        ticks += kmx_time() - t0;
        kinds[event.kind & 3u]++;
        if (event.kind != KM_EVENT_DONE) {
            break;
        }
    }
    w[0] = kinds[KM_EVENT_DONE];
    w[1] = kinds[KM_EVENT_TIMEOUT];
    w[2] = (uint32_t)ticks;
    return 0;
}

/* Step (d): windows, in mcycles over `repeats`, on core `core`'s reserved line. */
static uint32_t windows(const uint32_t* a, uint32_t* w) {
    const uint32_t repeats = a[1], core = a[2];
    const uint32_t xy = tile_of((int)core), x = xy & 0xFFFFu, y = xy >> 16;
    const uint32_t at = control_of() + KM_CONTROL_RESERVED;
    const uint32_t line[4] = {0, 0, 0, 0};
    noc_tlb_2m_t prog;
    uint64_t t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        noc_configure_tlb_2m_checked(SPARE_WINDOW, x, y, 0, NOC_ORDER_POSTED, &prog);
    }
    w[0] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        noc_configure_tlb_2m_checked(SPARE_WINDOW, x, y, 0, NOC_ORDER_POSTED, &prog);
        w[5] += (uint32_t)noc_verify_tlb_2m(SPARE_WINDOW, &prog) != 0;
    }
    w[1] = (uint32_t)(cycles() - t);
    volatile uint8_t* win = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            SPARE_WINDOW, x, y, 0, NOC_ORDER_POSTED, &prog
        )
    );
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 4, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"(line), "r"(win + at)
            : "memory", "v8"
        );
        x280_fence_ow();
    }
    w[2] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        volatile uint8_t* v = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                SPARE_WINDOW, x, y, 0, NOC_ORDER_POSTED, &prog
            )
        );
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 4, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"(line), "r"(v + at)
            : "memory", "v8"
        );
        x280_fence_ow();
    }
    w[3] = (uint32_t)(cycles() - t);
    noc_clear_tlb_2m(SPARE_WINDOW);
    /* how many windows a cluster can hold: every free one aimed and read back */
    uint32_t held = 0;
    for (uint32_t i = FREE_FIRST; i < FREE_END; i++) {
        if (
            i >= KMX_PORT_WINDOW && i < KMX_PORT_WINDOW + KMX_PORTS_CORES &&
            kmx_port_owner((int)(i - KMX_PORT_WINDOW)) >= 0
        ) {
            held++; /* the library's own, open for the run */
            continue;
        }
        if (
            noc_configure_tlb_2m_checked(i, x, y, 0, NOC_ORDER_POSTED, &prog) != 0 &&
            noc_verify_tlb_2m(i, &prog) == 0
        ) {
            held++;
        }
    }
    for (uint32_t i = FREE_FIRST; i < FREE_END; i++) {
        if (
            !(
                i >= KMX_PORT_WINDOW && i < KMX_PORT_WINDOW + KMX_PORTS_CORES &&
                kmx_port_owner((int)(i - KMX_PORT_WINDOW)) >= 0
            )
        ) {
            noc_clear_tlb_2m(i);
        }
    }
    w[4] = held;
    /* the serve loop's parts, each `repeats` times on core `core` */
    kmx_message m;
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        kmx_port_asked((int)core, &m);
    }
    w[6] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        kmx_port_told((int)core, &m);
    }
    w[7] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        kmx_stop_requested();
    }
    w[8] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        kmx_time();
    }
    w[9] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        kmx_port_tell((int)core, 0, 0, 0);
    }
    w[10] = (uint32_t)(cycles() - t);
    /* what a look at the mailbox is made of: a word, 16 bytes, a fence */
    const uint8_t* lim = (const uint8_t*)kmx_args_get();
    const volatile uint32_t*
        box = (
            (const volatile uint32_t*)(
                lim + ((const kmx_ports*)(lim + KMX_LIM_PORTS))->at + KMX_PORTS_BOX +
                64u * core
            )
        );
    uint32_t sink = 0, v4[4];
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        sink += box[0];
    }
    w[11] = (uint32_t)(cycles() - t);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 4, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"(box), "r"(v4)
            : "memory", "v8"
        );
        sink += v4[0];
    }
    w[12] = (uint32_t)(cycles() - t);
    /* a word of the image's own data, and of a region (cached DRAM) */
    static volatile uint32_t g_word __attribute__((section(".data"))) = 1;
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        sink += g_word;
    }
    w[13] = (uint32_t)(cycles() - t);
    const volatile uint32_t* region = (const volatile uint32_t*)kmx_region(0, 0);
    t = cycles();
    for (uint32_t i = 0; i < repeats; i++) {
        sink += region[0];
    }
    w[14] = (uint32_t)(cycles() - t);
    w[15] = sink & 1u;
    return 0;
}

int kmx_main(int cpu) {
    (void)cpu;
    const kmx_args* args = kmx_args_get();
    uint32_t a[XP_WORDS], w[48];
    for (uint32_t i = 0; i < XP_WORDS; i++) {
        a[i] = i < args->nwords ? args->words[i] : 0;
    }
    for (int i = 0; i < 48; i++) {
        w[i] = 0;
    }
    if (kmx_cluster() != kmx_lead()) {
        const int e = kmx_serve();
        finish(e == KMX_OK ? 0u : 3u, w, 16);
        return 0;
    }
    if (kmx_port_cores() == 0) {
        return 1;
    }
    uint32_t code = 2;
    if (a[0] == XP_MODE_TELL || a[0] == XP_MODE_SEND) {
        code = one_core(a, w);
    } else if (a[0] == XP_MODE_WINDOW) {
        code = windows(a, w);
    } else if (a[0] == XP_MODE_ASK || a[0] == XP_MODE_BLOCK || a[0] == XP_MODE_RATE) {
        code = all_cores(a, w);
    }
    finish(code, w, 16);
    return 0;
}
