/* kmx start-up and calls: every hart checks the argument block; hart 0 checks the
 * bank and runs kmx_main; harts 1 to 3 run kmx_par until it ends, stops or is late. */
#include <stdint.h>

#include "kmx.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"
#include "x280_hart_protocol.h"

/* The cluster's L2 cache controller: Flush64 writes back and invalidates the line
 * that holds the physical address stored into it. */
#define KMX_CCACHE_FLUSH64 0x02010200ull
#define KMX_POLL_EVERY 256u
/* Hart h's L2 prefetcher (BASIC_CONTROL, then USER_CONTROL) and the values the chip's
 * documentation recommends after reset: scalar code that walks memory runs faster. */
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

/* hart 0's kmx_par mailbox */
enum { PAR_GEN = 0, PAR_FN = 1, PAR_ARG = 2, PAR_EXIT = 3 };

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

static inline volatile uint64_t* par_box(void) {
    return (volatile uint64_t*)(lim() + KMX_LIM_PAR);
}

static inline volatile uint64_t* par_done(void) {
    return (volatile uint64_t*)(lim() + KMX_LIM_DONE);
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

void kmx_take(int id) {
    uint64_t bytes;
    void* p = kmx_region(id, &bytes);
    if (p) {
        flush((uint64_t)(uintptr_t)p, bytes);
    }
}

void kmx_give(int id) {
    kmx_take(id); /* Flush64 both writes back and drops a line */
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

int kmx_par(void (*fn)(int hart, void* arg), void* arg) {
    if (hart_id() != 0) {
        fail(KMX_ERR_PAR);
        return KMX_STOPPED;
    }
    volatile uint64_t* box = par_box();
    volatile uint64_t* done = par_done();
    const uint64_t gen = box[PAR_GEN] + 1;
    box[PAR_FN] = (uint64_t)(uintptr_t)fn;
    box[PAR_ARG] = (uint64_t)(uintptr_t)arg;
    x280_fence_ow();
    box[PAR_GEN] = gen;
    x280_fence_ow();
    fn(0, arg);
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (done[1] >= gen && done[2] >= gen && done[3] >= gen) {
            x280_fence_iorw();
            return KMX_OK;
        }
        if (spins % KMX_POLL_EVERY == 0) {
            const int end = must_end(0);
            if (end != KMX_OK) {
                fail(end == KMX_LATE ? KMX_ERR_LATE : KMX_ERR_STOPPED);
                return end;
            }
        }
    }
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

/* Harts 1 to 3: run each kmx_par call of hart 0 until it is done, a stop is asked
 * for, or the run's deadline passes. */
static void serve(uint64_t h) {
    volatile uint64_t* box = par_box();
    volatile uint64_t* done = par_done();
    uint64_t seen = 0;
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (box[PAR_EXIT] != 0) {
            return;
        }
        const uint64_t gen = box[PAR_GEN];
        if (gen > seen) {
            x280_fence_ir();
            void (*fn)(int, void*) = (void (*)(int, void*))(uintptr_t)box[PAR_FN];
            fn((int)h, (void*)(uintptr_t)box[PAR_ARG]);
            seen = gen;
            x280_fence_iorw();
            done[h] = gen;
            x280_fence_ow();
            continue;
        }
        if (spins % KMX_POLL_EVERY == 0 && must_end(h) != KMX_OK) {
            return;
        }
    }
}

/* Whether the argument block is this run's and every entry is in range. */
static int args_valid(const volatile kmx_args* a, uint64_t run) {
    if (x280_hart_args_bytes() < KMX_ARGS_BYTES || a->run_value != run ||
        a->magic != KMX_MAGIC || a->version != KMX_VERSION || a->deadline == 0 ||
        a->deadline > KMX_DEADLINE_MAX || a->nregions > KMX_REGIONS_MAX ||
        a->nwords > KMX_WORDS_MAX || (a->check & 63u)) {
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

/* Hart 0: the bank check, kmx_main, the result block. */
static uint64_t run_main(const volatile kmx_args* a) {
    volatile kmx_result_block* r = result();
    r->state = KMX_STATE_BOOT;
    x280_fence_ow();
    const volatile kmx_ring_desc* d = (
        (const volatile kmx_ring_desc*)(lim() + KMX_LIM_RING)
    );
    if (d->cmd != 0 && !kmx_ring_open(&g_ring, d)) {
        fail(KMX_ERR_ARGS); /* a ring is named but its line is out of range */
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
    const uint64_t ticks = x280_mtime() - g_start;
    r->ticks = (uint32_t)ticks;
    par_box()[PAR_EXIT] = 1;
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
    volatile uint32_t* pf = (
        (volatile uint32_t*)(uintptr_t)(KMX_L2PF_BASE + (hartid & 3u) * KMX_L2PF_STRIDE)
    );
    pf[0] = KMX_L2PF_BASIC;
    pf[1] = KMX_L2PF_USER;
    x280_fence_iorw();
    boot(hartid);
    return 0;
}
