/* kmx start-up and calls: every core checks the argument block; core 0 checks the
 * bank and runs kmx_main; cores 1 to 3 run what kmx_spawn gives them until it ends. */
#include <stdint.h>

#include "kmx.h"
#include "noc.h"
#include "rt/x280_atomic.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"
#include "x280_address_map.h"
#include "x280_hart_protocol.h"

/* The cluster's L2 cache controller: Flush64 writes back and invalidates the line
 * that holds the physical address stored into it. */
#define KMX_CCACHE_FLUSH64 0x02010200ull
/* The cluster cache's one way (WayEnable 0): 4 banks of 512 sets of 64-byte lines. The
 * cached Zero Device from its second 128 KiB, past the runtime's pinned input. */
#define KMX_CACHE_WAY_BYTES 0x20000ull
#define KMX_EVICT_PAD 0x0A020000ull
/* A fetch or store of 160 KiB or more evicts instead of sweeping: a sweep of its 2,560
 * lines costs 26.6 µs, an eviction of a cache full of dirty lines 26.4 µs. */
#define KMX_EVICT_FROM 0x28000ull
/* Read at the start (only read): ways enabled - 1, and each master's way mask (core
 * n's L2 is master 34 + n). */
#define KMX_CCACHE_WAYENABLE 0x02010008ull
#define KMX_CCACHE_WAYMASK 0x02010800ull
#define KMX_CCACHE_MASTER_CORE0 34ull
#define KMX_POLL_EVERY 256u
/* kmx_port_open looks for the event every KMX_PORT_LOOK calls on the lead's X280 core
 * 0, for a stop or the deadline every KMX_PORT_CHECK calls. */
#define KMX_PORT_LOOK 4u
#define KMX_PORT_CHECK 16u
/* Each core's L2 prefetcher (BASIC_CONTROL, USER_CONTROL) and the documented values,
 * set unless the host asks to keep them: scalar code that walks memory runs faster. */
#define KMX_L2PF_BASE 0x02030000ull
#define KMX_L2PF_STRIDE 0x2000ull
#define KMX_L2PF_BASIC 0x15811u
#define KMX_L2PF_USER 0x38c84eu
#define KMX_UNSET 0xFFFFFFFFu

static uint64_t g_deadline[4] __attribute__((section(".data"))) = {1, 1, 1, 1};
/* Set once core 0 serves commands: no core has a deadline from then on. A flag, not a
 * lifted deadline: a core that boots after core 0 stores its own deadline. */
static uint32_t g_serving __attribute__((section(".data"))) = 0;
static int must_end(uint64_t core);
static void flush(uint64_t address, uint64_t bytes);
static inline uint64_t core_id(void);

#define KMX_RING_FENCE_W() x280_fence_ow()
#define KMX_RING_FENCE_R() x280_fence_ir()
#define KMX_RING_SWEEP(address, bytes) flush((address), (bytes))
#define KMX_RING_NOW() x280_mtime()
#define KMX_RING_END() must_end(0)
#include "kmx_ring.h"

/* The ring, core 0's (a .data global: the image has no .bss). */
static kmx_ring_t g_ring __attribute__((section(".data"))) = {.next = 1};

/* Image globals live in .data (the image has no .bss): non-zero initialisers. */
static uint64_t g_start __attribute__((section(".data"))) = 1;
/* Set at the start when the cache is the one way kmx_evict empties: fetches and stores
 * of KMX_EVICT_FROM bytes or more then evict, else they sweep. */
static uint32_t g_one_way __attribute__((section(".data"))) = 0;
static uint32_t g_code __attribute__((section(".data"))) = KMX_UNSET;
static uint32_t g_in_parallel
    __attribute__((section(".data"))) = 0; /* core 0 in kmx_parallel */
/* kmx_parallel_all: the call number, whether every other cluster said it serves, and
 * each other cluster's LIM block through its window (0: none). */
static uint32_t g_all_gen __attribute__((section(".data"))) = 0;
static uint32_t g_all_ready __attribute__((section(".data"))) = 0;
static volatile uint8_t* g_peer[4] __attribute__((section(".data"))) = {0, 0, 0, 0};

static inline uint64_t core_id(void) {
    uint64_t core;
    __asm__ volatile("csrr %0, mhartid" : "=r"(core));
    return core;
}

static inline volatile uint8_t* lim(void) {
    return (volatile uint8_t*)(uintptr_t)x280_hart_args(core_id());
}

static inline volatile kmx_result_block* result(void) {
    return (volatile kmx_result_block*)(lim() + KMX_LIM_RESULT);
}

/* A core's spawn line (KMX_SPAWN_* words). */
static inline volatile uint64_t* spawn_line(uint64_t core) {
    return (volatile uint64_t*)(lim() + KMX_LIM_CORES + 64u * core);
}

static inline const volatile kmx_peers* peers(void) {
    return (const volatile kmx_peers*)(lim() + KMX_LIM_PEERS);
}

/* A kmx_parallel_all line (KMX_CALL_* words) of this cluster's LIM, or through the
 * window onto a cluster's. */
static inline volatile uint32_t* all_line(volatile uint8_t* block, uint32_t line) {
    return (volatile uint32_t*)(block + peers()->all + KMX_ALL_CALL + 64u * line);
}

/* Commands the first library error. */
static void fail(uint32_t error) {
    volatile kmx_result_block* block = result();
    if (block->error == KMX_ERROR_NONE) {
        block->error = error;
        x280_fence_ow();
    }
}

/* KMX_OK, or why the run must end now: a stop was asked for, or it is late. */
static int must_end(uint64_t core) {
    x280_fence_ir();
    const volatile uint32_t* stop = (const volatile uint32_t*)(lim() + KMX_LIM_STOP);
    if (
        *stop == KMX_STOP_WORD ||
        x280_hart_stop_requested(core, x280_hart_control(core)->run_value)
    ) {
        return KMX_STOPPED;
    }
    if (*(volatile const uint32_t*)&g_serving) {
        return KMX_OK;
    }
    const uint64_t deadline = ((volatile const uint64_t*)g_deadline)[core];
    return x280_mtime() > deadline ? KMX_LATE : KMX_OK;
}

/* Flush64 of every line of [address, address + bytes): a 4-byte store per line. */
static void flush(uint64_t address, uint64_t bytes) {
    volatile uint64_t* flush_register = (
        (volatile uint64_t*)(uintptr_t) KMX_CCACHE_FLUSH64
    );
    x280_fence_iorw();
    for (
        uint64_t line_address = address & ~63ull; line_address < address + bytes;
        line_address += 64
    ) {
        __asm__ volatile(".option push\n.option norvc\nsd %0, 0(%1)\n.option pop\n"
            :
            : "r"(line_address), "r"(flush_register)
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

int kmx_island(void) {
    return (int)(kmx_args_get()->island & 0xFFFFu);
}

int kmx_islands(void) {
    const uint32_t islands = kmx_args_get()->island >> 16;
    return islands == 0 ? 1 : (int)islands;
}

int kmx_island_of(uint32_t number) {
    const uint32_t island = (number - 1u) / 3u;
    return number >= 1u && island < (uint32_t)kmx_islands() ? (int)island : -1;
}

void* kmx_region(int id, uint64_t* bytes) {
    const kmx_args* arguments = kmx_args_get();
    if (id < 0 || (uint32_t)id >= arguments->nregions) {
        if (bytes) {
            *bytes = 0;
        }
        return 0;
    }
    if (bytes) {
        *bytes = arguments->region[id].bytes;
    }
    return (void*)(uintptr_t)(X280_LOCAL_DDR_TILE_BASE + arguments->region[id].offset);
}

/* A flush or invalidate sweep split by lines over the four cores: each core takes its
 * quarter. */
typedef struct {
    uint64_t address, lines;
} sweep_t;

static void sweep_part(int core, void* arg) {
    const sweep_t* range = (const sweep_t*)arg;
    const uint64_t per = (range->lines + 3) / 4, first = (uint64_t)core * per;
    if (first < range->lines) {
        const uint64_t count = range->lines - first < per ? range->lines - first : per;
        flush(range->address + first * 64, count * 64);
    }
}

/* Sweeps [address, address + bytes): on core 0 outside kmx_parallel over the four
 * cores. */
static void sweep(uint64_t address, uint64_t bytes) {
    if (core_id() != 0 || g_in_parallel) {
        flush(address, bytes); /* inside a kmx_parallel call: this core alone */
        return;
    }
    const uint64_t aligned_start = address & ~63ull;
    const uint64_t aligned_end = (address + bytes + 63) & ~63ull;
    sweep_t range = {aligned_start, (aligned_end - aligned_start) / 64};
    if (kmx_parallel(sweep_part, &range) != KMX_OK) {
        flush(address, bytes); /* busy, late or stopped: the flush is still whole */
    }
}

void kmx_invalidate_at(int id, uint64_t offset, uint64_t bytes) {
    uint64_t region_bytes;
    uint8_t* region = (uint8_t*)kmx_region(id, &region_bytes);
    if (region && offset < region_bytes) {
        sweep(
            (uint64_t)(uintptr_t)(region + offset),
            bytes < region_bytes - offset ? bytes : region_bytes - offset
        );
    }
}

void kmx_invalidate(int id) {
    kmx_invalidate_at(id, 0, ~0ull);
}

void kmx_flush(int id) {
    kmx_invalidate(id); /* Flush64 both writes back and drops a line */
}

void kmx_flush_at(int id, uint64_t offset, uint64_t bytes) {
    kmx_invalidate_at(id, offset, bytes);
}

void kmx_evict(void) {
    x280_fence_iorw();
    for (
        uint64_t line_address = KMX_EVICT_PAD;
        line_address < KMX_EVICT_PAD + KMX_CACHE_WAY_BYTES;
        line_address += 64
    ) {
        (void)*(volatile const uint64_t*)(uintptr_t)line_address;
    }
    x280_fence_iorw();
}

/* KMX_OK on core 0; elsewhere the error is recorded and the call rejected. */
static int ring_core(void) {
    if (core_id() != 0) {
        fail(KMX_ERROR_PARALLEL);
        return KMX_STOPPED;
    }
    return KMX_OK;
}

int kmx_submit(const km_command_t* command, uint32_t* handle) {
    const int status = ring_core();
    return status != KMX_OK ? status : kmx_ring_submit(&g_ring, command, handle);
}

int kmx_wait(uint32_t handle, uint64_t deadline, kmx_event* event) {
    int status = ring_core();
    if (status == KMX_OK) {
        status = kmx_ring_wait(&g_ring, handle, deadline, event);
    }
    if (status == KMX_LATE || (status == KMX_STOPPED && g_ring.commands != 0)) {
        fail(status == KMX_LATE ? KMX_ERROR_LATE : KMX_ERROR_STOPPED);
    }
    return status;
}

int kmx_poll(kmx_event* event) {
    const int status = ring_core();
    return status != KMX_OK ? status : kmx_ring_poll(&g_ring, event);
}

/* Commands the error of a wait that ended early and returns its code. */
static int ended(int end) {
    fail(end == KMX_LATE ? KMX_ERROR_LATE : KMX_ERROR_STOPPED);
    return end;
}

/* Core 0: KMX_OK at the dispatcher's next wake, the other events set aside for
 * kmx_poll; KMX_FULL with KMX_STASH set aside, or a stop or `deadline` (an error). */
static int wake(uint64_t deadline) {
    for (uint32_t spins = 1;; spins++) {
        kmx_event event;
        if (g_ring.stashed == KMX_STASH) {
            return KMX_FULL;
        }
        if (kmx_ring_take(&g_ring, &event)) {
            if (event.kind == KMX_EVENT_WAKE) {
                return KMX_OK;
            }
            g_ring.stash[g_ring.stashed++] = event;
        } else if (spins % KMX_RING_SPINS == 0) {
            const int end = must_end(0);
            if (end != KMX_OK || x280_mtime() > deadline) {
                return ended(end != KMX_OK ? end : KMX_LATE);
            }
        }
    }
}

/* Core 0: submits `command` (again after each busy rejection) and waits for its event
 * by `deadline`. KMX_OK once done; KMX_BAD when refused, timed out or stash full. */
static int until_done(
    const km_command_t* command, uint64_t deadline, kmx_event* event
) {
    int status = KMX_OK;
    for (;;) {
        uint32_t handle = 0;
        for (uint32_t spins = 1;; spins++) {
            status = kmx_ring_submit(&g_ring, command, &handle);
            if (status != KMX_FULL) {
                break;
            }
            if (spins % KMX_RING_SPINS == 0) {
                const int end = must_end(0);
                if (end != KMX_OK || x280_mtime() > deadline) {
                    return ended(end != KMX_OK ? end : KMX_LATE);
                }
            }
        }
        if (status == KMX_OK) {
            status = kmx_wait(handle, deadline, event);
        }
        if (status != KMX_OK) {
            return status == KMX_FULL ? KMX_BAD : status; /* KMX_STASH set aside */
        }
        if (event->kind == KM_EVENT_DONE) {
            return KMX_OK;
        }
        if (event->kind != KM_EVENT_REJECTED || event->auxiliary != KMX_REJECT_BUSY) {
            return KMX_BAD; /* refused, or its deadline's timeout event */
        }
        status = wake(deadline);
        if (status != KMX_OK) {
            return status == KMX_FULL ? KMX_BAD : status;
        }
    }
}

int kmx_load(
    uint32_t program,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) {
    const volatile kmx_ring_descriptor*
        descriptor = ((const volatile kmx_ring_descriptor*)(lim() + KMX_LIM_RING));
    kmx_event own;
    event = event != 0 ? event : &own;
    const int status = ring_core();
    if (status != KMX_OK) {
        return status;
    }
    if (program >= descriptor->programs) {
        return KMX_BAD;
    }
    km_command_t command;
    kmx_command(&command, program, KM_HANDLER_LOAD, first, count);
    command.a0 = descriptor->records + program * descriptor->record_step;
    return until_done(&command, deadline, event);
}

/* Core 0: the spawns started on each core (as its line's GEN word holds), and the
 * cores kmx_spawn started that no join has seen finish (one bit per core). */
static uint64_t g_spawned[4] __attribute__((section(".data"))) = {0, 0, 0, 0};
static uint32_t g_spawn_busy __attribute__((section(".data"))) = 0;

/* Cores [first_core, end_core): function and arg into each line, then each line's next
 * number. */
static void start(
    uint64_t first_core, uint64_t end_core, void (*function)(int, void*), void* arg
) {
    volatile uint64_t* line = spawn_line(0);
    for (uint64_t core = first_core; core < end_core; core++) {
        line[8 * core + KMX_SPAWN_FUNCTION] = (uint64_t)(uintptr_t)function;
        line[8 * core + KMX_SPAWN_ARG] = (uint64_t)(uintptr_t)arg;
    }
    x280_fence_ow();
    for (uint64_t core = first_core; core < end_core; core++) {
        line[8 * core + KMX_SPAWN_GEN] = ++g_spawned[core];
    }
    x280_fence_ow();
}

/* Whether one of cores [first_core, end_core) has not finished its last spawn. */
static int busy(uint64_t first_core, uint64_t end_core) {
    x280_fence_ir();
    for (uint64_t core = first_core; core < end_core; core++) {
        if (spawn_line(core)[KMX_SPAWN_DONE] < g_spawned[core]) {
            return 1;
        }
    }
    return 0;
}

int kmx_spawn(int core, void (*function)(int core, void* arg), void* arg) {
    if (core_id() != 0) {
        fail(KMX_ERROR_PARALLEL);
        return KMX_STOPPED;
    }
    if (core < 1 || core > 3) {
        return KMX_BAD;
    }
    if (busy((uint64_t)core, (uint64_t)core + 1)) {
        return KMX_FULL;
    }
    start((uint64_t)core, (uint64_t)core + 1, function, arg);
    g_spawn_busy |= 1u << core;
    return KMX_OK;
}

/* Core 0: waits until cores [first_core, end_core) finished their spawns. */
static int join(uint64_t first_core, uint64_t end_core) {
    const volatile uint64_t* line = spawn_line(0);
    const uint64_t spawned_1 = g_spawned[1];
    const uint64_t spawned_2 = g_spawned[2];
    const uint64_t spawned_3 = g_spawned[3];
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (
            (first_core > 1 || line[8 + KMX_SPAWN_DONE] >= spawned_1) &&
            (
                first_core > 2 || end_core < 3 || line[16 + KMX_SPAWN_DONE] >= spawned_2
            ) &&
            (end_core < 4 || line[24 + KMX_SPAWN_DONE] >= spawned_3)
        ) {
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

int kmx_join(int core) {
    if (core_id() != 0) {
        fail(KMX_ERROR_PARALLEL);
        return KMX_STOPPED;
    }
    if (core < 1 || core > 3) {
        return KMX_BAD;
    }
    const int status = join((uint64_t)core, (uint64_t)core + 1);
    if (status == KMX_OK) {
        g_spawn_busy &= ~(1u << core);
    }
    return status;
}

int kmx_parallel(void (*function)(int core, void* arg), void* arg) {
    if (core_id() != 0 || g_in_parallel) {
        fail(KMX_ERROR_PARALLEL); /* another core, or a kmx_parallel inside one */
        return KMX_STOPPED;
    }
    if (g_spawn_busy) {
        if (busy(1, 4)) {
            return KMX_FULL; /* a spawned core has not finished */
        }
        g_spawn_busy = 0;
    }
    g_in_parallel = 1;
    start(1, 4, function, arg);
    function(0, arg);
    const int status = join(1, 4);
    g_in_parallel = 0;
    return status;
}

uint32_t kmx_clusters(void) {
    const volatile kmx_peers* group = peers();
    return group->lead ? group->mask : 1u << kmx_cluster();
}

int kmx_lead(void) {
    const uint32_t lead = peers()->lead;
    return lead ? (int)lead : kmx_cluster();
}

/* 32-bit stores of a 64-bit value through a window, low half first. */
static void put64(volatile uint32_t* window, uint64_t value) {
    window[0] = (uint32_t)value;
    window[1] = (uint32_t)(value >> 32);
}

static uint64_t get64(const volatile uint32_t* window) {
    return (uint64_t)window[0] | (uint64_t)window[1] << 32;
}

/* This cluster's done line in the lead's LIM: number `generation` finished (0: it
 * serves). */
static void tell_lead(uint32_t generation) {
    volatile uint32_t* done_line = (
        all_line(g_peer[peers()->lead], (uint32_t)kmx_cluster())
    );
    if (generation == 0) {
        done_line[KMX_CALL_GEN] = 0;
        x280_fence_ow();
        done_line[KMX_CALL_NONCE] = peers()->nonce;
        x280_fence_ow();
    }
    done_line[KMX_CALL_GEN] = generation;
    x280_fence_ow();
}

/* Lead: waits until the cluster's done line holds this load's nonce and `generation`.
 */
static int wait_done(uint32_t cluster, uint32_t generation) {
    const volatile uint32_t* done_line = all_line(lim(), cluster);
    const uint32_t nonce = peers()->nonce;
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (
            done_line[KMX_CALL_NONCE] == nonce && done_line[KMX_CALL_GEN] >= generation
        ) {
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

/* Lead: number `generation` of the call line into the cluster's LIM (exit: the last).
 */
static void call(
    uint32_t cluster,
    uint32_t generation,
    uint64_t function,
    uint64_t arg,
    uint32_t exit
) {
    volatile uint32_t* call_line = all_line(g_peer[cluster], 0);
    put64(call_line + KMX_CALL_FUNCTION, function);
    put64(call_line + KMX_CALL_ARG, arg);
    call_line[KMX_CALL_EXIT] = exit;
    call_line[KMX_CALL_NONCE] = peers()->nonce;
    x280_fence_ow();
    call_line[KMX_CALL_GEN] = generation;
    x280_fence_ow();
}

/* What every core of a cluster runs for one kmx_parallel_all call. */
typedef struct {
    void (*function)(int cluster, int core, uint64_t arg);
    uint64_t arg;
    int cluster;
} all_call_t;

static void all_part(int core, void* arg) {
    const all_call_t* all_call = (const all_call_t*)arg;
    all_call->function(all_call->cluster, core, all_call->arg);
}

/* The other clusters of the lead's group, as bits. */
static uint32_t others(void) {
    const volatile kmx_peers* group = peers();
    return group->lead ? group->mask & ~(1u << kmx_cluster()) : 0;
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
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        const int status = (rest >> cluster & 1u) ? wait_done(cluster, 0) : KMX_OK;
        if (status != KMX_OK) {
            return status;
        }
    }
    g_all_ready = 1;
    return KMX_OK;
}

/* The lead's X280 core 0, not nested: KMX_OK, or KMX_STOPPED (library error) elsewhere.
 */
static int all_lead(void) {
    if (core_id() != 0 || g_in_parallel || kmx_lead() != kmx_cluster()) {
        fail(KMX_ERROR_PARALLEL);
        return KMX_STOPPED;
    }
    return KMX_OK;
}

/* kmx_parallel_all with `part` (given an all_call_t) as the lead cluster's kmx_parallel
 * part. */
static int all_run(
    void (*function)(int, int, uint64_t), uint64_t arg, void (*part)(int, void*)
) {
    const uint32_t rest = others();
    int status = all_ready();
    if (status != KMX_OK) {
        return status;
    }
    const uint32_t generation = ++g_all_gen;
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if (rest >> cluster & 1u) {
            call(cluster, generation, (uint64_t)(uintptr_t)function, arg, 0);
        }
    }
    all_call_t mine = {function, arg, kmx_cluster()};
    status = kmx_parallel(part, &mine);
    for (uint32_t cluster = 1; cluster < 4 && status == KMX_OK; cluster++) {
        status = (rest >> cluster & 1u) ? wait_done(cluster, generation) : KMX_OK;
    }
    return status;
}

int kmx_parallel_all(
    void (*function)(int cluster, int core, uint64_t arg), uint64_t arg
) {
    const int status = all_lead();
    return status != KMX_OK ? status : all_run(function, arg, all_part);
}

int kmx_serve(void) {
    if (core_id() != 0 || g_in_parallel) {
        fail(KMX_ERROR_PARALLEL);
        return KMX_STOPPED;
    }
    if (kmx_lead() == kmx_cluster()) {
        return KMX_NONE;
    }
    const volatile uint32_t* call_line = all_line(lim(), 0);
    const uint32_t nonce = peers()->nonce;
    uint32_t seen = 0;
    tell_lead(0);
    for (uint32_t spins = 1;; spins++) {
        x280_fence_ir();
        const uint32_t generation = call_line[KMX_CALL_GEN];
        if (generation == seen + 1 && call_line[KMX_CALL_NONCE] == nonce) {
            x280_fence_ir();
            if (call_line[KMX_CALL_EXIT]) {
                return KMX_OK;
            }
            all_call_t mine = {
                (void (*)(int, int, uint64_t))(uintptr_t)
                    get64(call_line + KMX_CALL_FUNCTION),
                get64(call_line + KMX_CALL_ARG),
                kmx_cluster()
            };
            seen = generation;
            g_all_ready = 1;
            const int status = kmx_parallel(all_part, &mine);
            if (status != KMX_OK) {
                return status;
            }
            tell_lead(generation);
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

/* Region `id` of the cluster's thread: its bank address and size, from the argument
 * block in its LIM (a read through the window); 0 for an unknown one. */
static uint64_t region_of(uint32_t cluster, int id, uint64_t* bytes) {
    *bytes = 0;
    const volatile uint8_t*
        block = (cluster == (uint32_t)kmx_cluster() ? lim() : g_peer[cluster]);
    if (block == 0 || id < 0) {
        return 0;
    }
    const volatile uint32_t* argument_block = (const volatile uint32_t*)block;
    const uint32_t region_count = argument_block[offsetof(kmx_args, nregions) / 4];
    if ((uint32_t)id >= region_count || region_count > KMX_REGIONS_MAX) {
        return 0;
    }
    const volatile uint32_t*
        entry = (argument_block + (KMX_ARGS_REGIONS + 16u * (uint32_t)id) / 4);
    *bytes = get64(entry + 2);
    return get64(entry);
}

/* 64-byte multiples of `source` stored through a window, 512 bytes per vector store. */
static void copy_out(
    volatile uint8_t* destination, const uint8_t* source, uint64_t bytes
) {
    while (bytes != 0) {
        uint64_t vector_length;
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetvli %0, %1, e64, m8, ta, ma\n"
            "vle64.v v0, (%2)\n"
            "vse64.v v0, (%3)\n"
            ".option pop\n"
            : "=&r"(vector_length)
            : "r"(bytes / 8), "r"(source), "r"(destination)
            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
        );
        source += vector_length * 8;
        destination += vector_length * 8;
        bytes -= vector_length * 8;
    }
}

/* One core's push of [source, source + bytes) to bank address `address` of DRAM tile
 * `tile`: posted stores per 2 MiB, the last line again, acked. 0, or 1 on a failure. */
static int push_range(
    uint64_t core,
    uint32_t tile,
    uint64_t address,
    const uint8_t* source,
    uint64_t bytes
) {
    const uint32_t x = tile & 0xFFFFu, y = tile >> 16;
    const uint32_t window = KMX_PUSH_WINDOW + (uint32_t)core;
    const uint64_t end = address + bytes;
    noc_tlb_2m_t programmed;
    while (address < end) {
        const uint64_t room = (
            NOC_2M_WINDOW_STRIDE - (address & (NOC_2M_WINDOW_STRIDE - 1u))
        );
        const uint64_t chunk = end - address < room ? end - address : room;
        volatile uint8_t* mapped = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                window, x, y, address, NOC_ORDER_POSTED, &programmed
            )
        );
        if (mapped == 0) {
            return 1;
        }
        copy_out(mapped, source, chunk);
        x280_fence_ow();
        address += chunk;
        source += chunk;
    }
    /* The last line again through a non-posted window: its acknowledgement comes
     * after the posted stores before it on the same path. */
    volatile uint32_t* last_line = (
        (volatile uint32_t*)noc_configure_tlb_2m_checked(
            window, x, y, end - 64, NOC_ORDER_DEFAULT, &programmed
        )
    );
    if (last_line == 0) {
        return 1;
    }
    const volatile uint32_t* source_line = (const volatile uint32_t*)(source - 64);
    for (int word = 0; word < 16; word++) {
        last_line[word] = source_line[word];
    }
    x280_fence_ow();
    noc_clear_tlb_2m(window);
    return 0;
}

typedef struct {
    uint32_t tile;
    uint64_t address, bytes;
    const uint8_t* source;
    uint32_t bad[4];
} push_t;

/* A core's quarter, in whole lines. */
static void push_part(int core, void* arg) {
    push_t* push = (push_t*)arg;
    const uint64_t lines = push->bytes / 64, per = (lines + 3) / 4;
    const uint64_t first = (uint64_t)core * per;
    push->bad[core] = 0;
    if (first < lines) {
        const uint64_t chunk_bytes = (lines - first < per ? lines - first : per) * 64;
        push->bad[core] = (
            (uint32_t)push_range(
                (uint64_t)core,
                push->tile,
                push->address + first * 64,
                push->source + first * 64,
                chunk_bytes
            )
        );
    }
}

int kmx_push(int cluster, int id, uint64_t offset, const void* source, uint64_t bytes) {
    const uint32_t peer = (uint32_t)cluster;
    if (
        cluster < 1 || cluster > 3 || !(kmx_clusters() >> peer & 1u) ||
        ((offset | bytes | (uint64_t)(uintptr_t)source) & 63u)
    ) {
        return KMX_BAD;
    }
    if (peer != (uint32_t)kmx_cluster()) {
        const int status = all_ready(); /* its argument block is this load's */
        if (status != KMX_OK) {
            return status;
        }
    }
    uint64_t region_bytes;
    const uint64_t region_address = region_of(peer, id, &region_bytes);
    if (region_bytes == 0 || offset > region_bytes || bytes > region_bytes - offset) {
        return KMX_BAD;
    }
    if (bytes == 0) {
        return KMX_OK;
    }
    push_t push = {
        peers()->bank[peer],
        region_address + offset,
        bytes,
        (const uint8_t*)source,
        {0, 0, 0, 0}
    };
    const uint64_t core = core_id();
    if (core == 0 && !g_in_parallel && kmx_parallel(push_part, &push) == KMX_OK) {
        return push.bad[0] | push.bad[1] | push.bad[2] | push.bad[3] ? KMX_BAD : KMX_OK;
    }
    return (
        push_range(core, push.tile, push.address, push.source, bytes) ? KMX_BAD : KMX_OK
    );
}

/* X280 ports per core: KM_CONTROL_X280 through its window, serving X280 core + 1, last
 * ask answered, tell seen, tell sent; few words read (a LIM load costs ~30 cycles). */
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
    const volatile kmx_ports* ports_line = ports();
    const uint32_t area_offset = ports_line->at;
    const uint32_t control = ports_line->control;
    const uint32_t cores = ports_line->cores;
    if (area_offset == 0) {
        return 1;
    }
    if (
        (area_offset & 63u) || area_offset < x280_hart_args_bytes() ||
        (uint64_t)(uintptr_t)lim() + area_offset + KMX_PORTS_BYTES > X280_LIM_END ||
        cores == 0 || cores > KMX_PORTS_CORES || (control & 63u) ||
        control + KM_CONTROL_BYTES > KMX_L1_BYTES ||
        (ports_line->flags & ~KMX_PORTS_NOC1)
    ) {
        return 0;
    }
    const volatile uint8_t* area = lim() + area_offset;
    const uint32_t me = (uint32_t)kmx_cluster();
    for (uint32_t core = 0; core < cores; core++) {
        const uint32_t owner = area[KMX_PORTS_OWNER + core];
        if (owner >> 2 != me) {
            continue;
        }
        const uint32_t tile = (
            ((const volatile uint32_t*)(area + KMX_PORTS_TILES))[core]
        );
        const uint32_t x = tile & 0xFFFFu, y = tile >> 16;
        if (x > NOC_MAX_COORD || y > NOC_MAX_COORD) {
            return 0;
        }
        noc_tlb_2m_t programmed;
        programmed.data
            [0] = (programmed.data[1] = programmed.data[2] = programmed.data[3] = 0);
        noc_fill_tlb_2m(&programmed, x, y, x, y, 0, NOC_ORDER_POSTED, 0, 0u);
        programmed.noc_sel = (ports_line->flags & KMX_PORTS_NOC1) ? 1u : 0u;
        volatile uint8_t* mapped = (
            (volatile uint8_t*)noc_configure_tlb_2m_ext(
                KMX_PORT_WINDOW + core, &programmed, 0
            )
        );
        g_port[core] = mapped + control + KM_CONTROL_X280;
        g_port_who[core] = (owner & 3u) + 1u;
        if (noc_verify_tlb_2m(KMX_PORT_WINDOW + core, &programmed) != 0) {
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
    for (uint32_t core = 0; core < KMX_PORTS_CORES; core++) {
        if (g_port[core] != 0) {
            noc_clear_tlb_2m(KMX_PORT_WINDOW + core);
            g_port[core] = 0;
        }
        g_port_who[core] = 0;
    }
    g_port_cores = 0;
}

/* Whether the calling X280 core serves `core`. */
static inline int port_mine(int core) {
    return (uint32_t)core < KMX_PORTS_CORES && g_port_who[core] == core_id() + 1u;
}

/* One 16-byte store of {sequence, first, second, third} into line `offset` of the
 * core's KM_CONTROL_X280. */
static void port_store(int core, uint32_t offset, const uint32_t* line) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(line), "r"(g_port[core] + offset)
        : "memory", "v8"
    );
    x280_fence_ow();
}

/* Line `offset` of the core's mailbox as one 16-byte load, and its sequence again: the
 * two must agree (the core's 16 bytes land whole). Torn on every try: no message. */
static void box_read(int core, uint32_t offset, kmx_message* message) {
    const volatile uint32_t* box_line = (
        (const volatile uint32_t*)(g_box + 64u * core + offset)
    );
    for (int attempt = 0; attempt < 4; attempt++) {
        x280_fence_ir();
        __asm__ volatile(
            ".option push\n.option arch, +v\n"
            "vsetivli zero, 4, e32, m1, ta, ma\n"
            "vle32.v v8, (%0)\n"
            "vse32.v v8, (%1)\n"
            ".option pop\n"
            :
            : "r"(box_line), "r"(message)
            : "memory", "v8"
        );
        if (box_line[0] == message->sequence) {
            return;
        }
    }
    message->sequence = 0;
}

int kmx_port_cores(void) {
    return (int)g_port_cores;
}

int kmx_port_owner(int core) {
    return (uint32_t)core < KMX_PORTS_CORES ? (int)g_port_who[core] - 1 : -1;
}

int kmx_port_tell(int core, uint32_t first, uint32_t second, uint32_t third) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    const uint32_t line[4] = {++g_tells[core], first, second, third};
    port_store(core, KM_X280_TELL, line);
    return KMX_OK;
}

int kmx_port_asked(int core, kmx_message* message) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    box_read(core, KM_BOX_ASK, message);
    return (
        message->sequence != 0 && message->sequence != g_answered[core]
            ? KMX_OK
            : KMX_NONE
    );
}

int kmx_port_answer(
    int core, uint32_t sequence, uint32_t first, uint32_t second, uint32_t third
) {
    if (!port_mine(core) || sequence == 0) {
        return KMX_BAD;
    }
    const uint32_t line[4] = {sequence, first, second, third};
    port_store(core, KM_X280_ANSWER, line);
    g_answered[core] = sequence;
    return KMX_OK;
}

int kmx_port_told(int core, kmx_message* message) {
    if (!port_mine(core)) {
        return KMX_BAD;
    }
    box_read(core, KM_BOX_TELL, message);
    if (message->sequence == 0 || message->sequence == g_told[core]) {
        return KMX_NONE;
    }
    g_told[core] = message->sequence;
    return KMX_OK;
}

/* kmx_port_run on the lead's X280 core 0: inside one, its handle, the run's number,
 * whether it ended and why (KMX_OK: its event, in g_port_event). */
static uint32_t g_port_live __attribute__((section(".data"))) = 0;
static uint32_t g_port_handle __attribute__((section(".data"))) = 0;
static uint32_t g_port_run __attribute__((section(".data"))) = 0;
static uint32_t g_port_ended __attribute__((section(".data"))) = 0;
static int g_port_why __attribute__((section(".data"))) = KMX_OK;
static kmx_event g_port_event __attribute__((section(".data"))) = {0, 0, 0, 0};
static uint32_t g_open_calls[4] __attribute__((section(".data"))) = {0, 0, 0, 0};

/* The run state into the ports line of every cluster of the group. */
static void port_state(uint32_t value) {
    const uint32_t offset = KMX_LIM_PORTS + offsetof(kmx_ports, state);
    *(volatile uint32_t*)(lim() + offset) = value;
    const uint32_t rest = others();
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if ((rest >> cluster & 1u) && g_peer[cluster] != 0) {
            *(volatile uint32_t*)(g_peer[cluster] + offset) = value;
        }
    }
    x280_fence_ow();
}

/* The lead's X280 core 0: the run ends, for `why`, on every X280 core of the group. */
static void port_end(int why) {
    g_port_why = why;
    g_port_ended = 1;
    port_state(g_port_run << 1 | 1u);
}

int kmx_port_open(void) {
    const uint64_t core = core_id();
    const uint32_t call_count = ++g_open_calls[core];
    const uint32_t check = call_count % KMX_PORT_CHECK == 0;
    if (core == 0 && g_port_live) {
        if (g_port_ended) {
            return 0;
        }
        if (call_count % KMX_PORT_LOOK != 0) {
            return 1;
        }
        const int status = kmx_ring_look(&g_ring, g_port_handle, &g_port_event);
        if (status != KMX_NONE) {
            port_end(status);
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
    if (state == 0 || (state & 1u) || (check && must_end(core) != KMX_OK)) {
        return 0;
    }
    return 1;
}

/* The lead cluster's part of a kmx_port_run: the function, then on X280 core 0 the end.
 */
static void port_part(int x280_core, void* arg) {
    all_part(x280_core, arg);
    while (x280_core == 0 && kmx_port_open()) {
        ;
    }
}

int kmx_port_run(
    const km_command_t* command,
    void (*function)(int cluster, int x280_core, uint64_t arg),
    uint64_t arg,
    kmx_event* event
) {
    int status = all_lead();
    if (status != KMX_OK) {
        return status;
    }
    if (g_port_cores == 0) {
        return KMX_BAD;
    }
    status = all_ready();
    if (status == KMX_OK && g_spawn_busy && busy(1, 4)) {
        return KMX_FULL; /* a spawned core still runs: nothing submitted */
    }
    uint32_t handle = 0;
    status = status == KMX_OK ? kmx_submit(command, &handle) : status;
    if (status != KMX_OK) {
        return status;
    }
    g_port_handle = handle;
    g_port_ended = 0;
    g_port_why = KMX_OK;
    g_port_live = 1;
    port_state(++g_port_run << 1);
    status = all_run(function, arg, port_part);
    g_port_live = 0;
    if (!g_port_ended) {
        port_end(status);
    }
    if (status != KMX_OK) {
        return status;
    }
    if (g_port_why == KMX_LATE || g_port_why == KMX_STOPPED) {
        return ended(g_port_why);
    }
    if (g_port_why == KMX_OK) {
        *event = g_port_event;
    }
    return g_port_why;
}

/* The pull's line in this cluster's (or a peer's) ports line, by word. */
#define KMX_PULL_AT(field) (KMX_LIM_PORTS + offsetof(kmx_ports, field))

/* This cluster's place among the group's clusters that serve at least one core of the
 * run (the ports area's owner table), and how many there are. */
static uint32_t group_place(uint32_t* group_count) {
    const volatile kmx_ports* ports_line = ports();
    const volatile uint8_t* area = lim() + ports_line->at;
    uint32_t serving = 0;
    for (uint32_t core = 0; ports_line->at != 0 && core < g_port_cores; core++) {
        serving |= 1u << (area[KMX_PORTS_OWNER + core] >> 2 & 3u);
    }
    const uint32_t mask = kmx_clusters() & serving, me = (uint32_t)kmx_cluster();
    uint32_t place = 0;
    *group_count = 0;
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        place += (mask >> cluster & 1u) && cluster < me;
        *group_count += mask >> cluster & 1u;
    }
    return place;
}

/* Each X280 core: answers its cores' pulls from the counter, `pick` or the log. */
static void pull_part(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)arg;
    const volatile kmx_ports* ports_line = ports();
    const uint32_t units = ports_line->pull[0];
    const uint32_t per = ports_line->pull[1];
    const uint32_t mode = ports_line->pull[2] & 0xFFu;
    const uint64_t pick_address = ports_line->pick;
    const uint32_t with_word = ports_line->pull[3];
    const uint64_t pick_argument = ports_line->arg;
    const uint64_t ranges = ((uint64_t)units + per - 1u) / per;
    volatile uint32_t* next = (volatile uint32_t*)(lim() + KMX_PULL_AT(next));
    uint64_t bytes = 0;
    uint32_t* log = (
        mode != KMX_PULL_COUNT ? kmx_region((int)(ports_line->pull[2] >> 8), &bytes) : 0
    );
    /* a cluster without the log region (or under one range a core): every ask lost */
    const uint32_t no_log = (
        mode != KMX_PULL_COUNT && (log == 0 || bytes < KMX_PULL_LOG_BYTES(1))
    );
    const uint32_t capacity = (
        log && !no_log ? (uint32_t)(bytes / 4u / KMX_PORTS_CORES) - 1u : 0
    );
    uint32_t* lists = log ? log + KMX_PORTS_CORES : 0;
    uint32_t groups;
    const uint32_t place = group_place(&groups);
    uint8_t mine[KMX_PORTS_CORES];
    uint32_t given[KMX_PORTS_CORES];
    int served_count = 0;
    for (int core = 0; core < kmx_port_cores(); core++) {
        if (kmx_port_owner(core) == x280_core) {
            given[served_count] = 0;
            mine[served_count++] = (uint8_t)core;
            if (mode == KMX_PULL_REPLAY && !no_log) { /* the host's or a log run's */
                flush((uint64_t)(uintptr_t)(log + core), 4);
                const uint32_t logged = log[core] < capacity ? log[core] : capacity;
                flush((uint64_t)(uintptr_t)(lists + core * capacity), 4ull * logged);
            }
        }
    }
    uint32_t lost = 0;
    while (kmx_port_open()) {
        for (int served = 0; served < served_count; served++) {
            const int core = mine[served];
            kmx_message message;
            if (kmx_port_asked(core, &message) != KMX_OK) {
                continue;
            }
            uint64_t range_index = KMX_PULL_NONE;
            uint32_t word = 0;
            if (message.w[0] != KM_PULL_ASK) {
            } else if (no_log) {
                lost++;
            } else if (mode == KMX_PULL_REPLAY) {
                range_index = (
                    given[served] < log[core] && given[served] < capacity
                        ? lists[core * capacity + given[served]]
                        : KMX_PULL_NONE
                );
            } else if (pick_address != 0 && with_word) {
                const uint64_t picked = (
                    (
                        (uint64_t (*)(int, uint32_t, uint64_t))(uintptr_t) pick_address
                    )(core, given[served], pick_argument)
                );
                range_index = (uint32_t)picked;
                word = (uint32_t)(picked >> 32);
            } else if (pick_address != 0) {
                range_index = (
                    (
                        (uint32_t (*)(int, uint32_t, uint64_t))(uintptr_t) pick_address
                    )(core, given[served], pick_argument)
                );
            } else {
                range_index = (uint64_t)x280_amoadd_w(next, 1u) * groups + place;
            }
            uint32_t first = 0, count = 0;
            if (range_index == KMX_PULL_NONE) {
                word = 0;
            }
            if (range_index < ranges) {
                first = (uint32_t)range_index * per;
                count = units - first < per ? units - first : per;
                if (mode == KMX_PULL_LOG && given[served] < capacity) {
                    lists[core * capacity + given[served]] = (uint32_t)range_index;
                } else if (mode == KMX_PULL_LOG) {
                    lost++;
                }
                given[served]++;
            }
            kmx_port_answer(core, message.sequence, first, count, count ? word : 0);
        }
    }
    for (
        int served = 0; served < served_count && mode == KMX_PULL_LOG && !no_log;
        served++
    ) {
        log[mine[served]] = given[served];
        flush((uint64_t)(uintptr_t)(log + mine[served]), 4);
        const uint32_t logged = given[served] < capacity ? given[served] : capacity;
        flush((uint64_t)(uintptr_t)(lists + mine[served] * capacity), 4ull * logged);
    }
    if (lost != 0) {
        x280_amoadd_w((volatile uint32_t*)(lim() + KMX_PULL_AT(lost)), lost);
    }
}

/* The lead: the pull's words into the ports line of a cluster (its block), the counter
 * and the lost count zeroed. */
static void pull_line(volatile uint8_t* block, const kmx_pull* settings) {
    volatile uint32_t* words = (volatile uint32_t*)(block + KMX_LIM_PORTS);
    words[offsetof(kmx_ports, next) / 4] = 0;
    words[offsetof(kmx_ports, lost) / 4] = 0;
    words[offsetof(kmx_ports, pull) / 4] = settings->units;
    words[offsetof(kmx_ports, pull) / 4 + 1] = settings->per;
    words[offsetof(kmx_ports, pull) / 4 + 2] = settings->mode | settings->log << 8;
    words[offsetof(kmx_ports, pull) / 4 + 3] = settings->pick_word != 0;
    put64(
        words + offsetof(kmx_ports, pick) / 4,
        settings->pick_word != 0
            ? (uint64_t)(uintptr_t)settings->pick_word
            : (uint64_t)(uintptr_t)settings->pick
    );
    put64(words + offsetof(kmx_ports, arg) / 4, settings->arg);
}

int kmx_pull_run(
    const km_command_t* command, const kmx_pull* settings, kmx_event* event
) {
    int status = all_lead();
    if (status != KMX_OK) {
        return status;
    }
    uint64_t bytes = 0;
    if (
        settings->mode > KMX_PULL_REPLAY || settings->log > 255u ||
        settings->per == 0 || (settings->pick != 0 && settings->pick_word != 0) ||
        (
            settings->mode != KMX_PULL_COUNT &&
            (
                kmx_region((int)settings->log, &bytes) == 0 ||
                bytes < KMX_PULL_LOG_BYTES(1)
            )
        )
    ) {
        return KMX_BAD;
    }
    status = all_ready();
    if (status != KMX_OK) {
        return status;
    }
    const uint32_t rest = others();
    pull_line(lim(), settings);
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if ((rest >> cluster & 1u) && g_peer[cluster] != 0) {
            pull_line(g_peer[cluster], settings);
        }
    }
    x280_fence_ow();
    status = kmx_port_run(command, pull_part, 0, event);
    uint32_t lost = *(volatile uint32_t*)(lim() + KMX_PULL_AT(lost));
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if ((rest >> cluster & 1u) && g_peer[cluster] != 0) {
            lost += *(volatile uint32_t*)(g_peer[cluster] + KMX_PULL_AT(lost));
        }
    }
    return status == KMX_OK && lost != 0 ? KMX_BAD : status;
}

uint64_t kmx_time(void) {
    return x280_mtime();
}

int kmx_stop_requested(void) {
    return must_end(core_id()) != KMX_OK;
}

void kmx_progress(uint32_t where) {
    volatile kmx_result_block* block = result();
    block->where = where;
    block->t_where = (uint32_t)(x280_mtime() - g_start);
    x280_fence_ow();
}

void kmx_command(
    km_command_t* command,
    uint32_t program,
    uint32_t handler,
    uint32_t first,
    uint32_t count
) {
    /* field by field: a struct assignment may call memset, which does not link */
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = (handler & 0xFFu) | program << 24;
    command->group = (first & 0xFFFFu) | count << 16;
    command->a0 = command->a1 = command->a2 = command->a3 = command->a4 = 0;
    command->a5 = command->a6 = command->a7 = command->a8 = command->a9 = 0;
}

int kmx_block(km_command_t* command, int id, uint64_t offset, uint32_t bytes) {
    uint64_t region_bytes;
    const uint64_t address = region_of((uint32_t)kmx_cluster(), id, &region_bytes);
    if (
        region_bytes == 0 || bytes == 0 || bytes > KM_BLOCK_BYTES ||
        offset % 64u != 0 || offset > region_bytes ||
        (bytes + 63u) / 64u * 64u > region_bytes - offset ||
        address + offset > UINT32_MAX
    ) {
        return KMX_BAD;
    }
    command->a7 = (uint32_t)(address + offset);
    command->a8 = (
        bytes |
        (KMX_BANK_OF_CLUSTER((uint32_t)kmx_cluster()) + 1u) << KM_BLOCK_BANK_SHIFT
    );
    return KMX_OK;
}

/* A copy between region `id` at `offset` and `other`, its command filled: KMX_OK, or
 * KMX_BAD for a bound (km_copy_fits on the library's side, with the region's size). */
#define KMX_DRAM_BANKS 8u /* the p150a's DRAM banks */

static int copy_command(
    km_command_t* command,
    int id,
    uint64_t offset,
    const kmx_dram* other,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    int fetch
) {
    const volatile kmx_ring_descriptor*
        descriptor = ((const volatile kmx_ring_descriptor*)(lim() + KMX_LIM_RING));
    const kmx_args* arguments = kmx_args_get();
    if (
        id < 0 || (uint32_t)id >= arguments->nregions || other == 0 ||
        descriptor->buffer == 0 || bytes == 0 || ((offset | bytes) % KM_COPY_ALIGN)
    ) {
        return KMX_BAD;
    }
    const kmx_region_t* region = &arguments->region[id];
    uint32_t page = other->page;
    if (page == 0 && other->bank != KM_COPY_INTERLEAVED) {
        page = KM_TRANSFER_CHUNK_BYTES; /* a range's: the largest power of two */
        while (bytes % page) {
            page >>= 1;
        }
    }
    if (
        offset > region->bytes || bytes > region->bytes - offset || page == 0 ||
        page > KM_TRANSFER_CHUNK_BYTES || bytes % page || bytes / page > 0xFFFFFFFFull
    ) {
        return KMX_BAD;
    }
    /* field by field: a struct assignment may call memcpy, which does not link */
    km_copy_t copy;
    km_copy_side_t* mine = fetch ? &copy.to : &copy.from;
    km_copy_side_t* theirs = fetch ? &copy.from : &copy.to;
    mine->address = (uint32_t)(region->offset + offset);
    mine->bank = KMX_BANK_OF_CLUSTER((uint32_t)kmx_cluster());
    mine->first = 0;
    theirs->address = other->address;
    theirs->bank = other->bank;
    theirs->first = other->first;
    copy.page = page;
    copy.pages = (uint32_t)(bytes / page);
    copy.buffer = descriptor->buffer;
    copy.chunk = KM_TRANSFER_CHUNK_BYTES / page * page;
    if (
        region->offset + region->bytes > KM_COPY_BANK_END ||
        !km_copy_fits(&copy, count, KMX_L1_BYTES, KMX_DRAM_BANKS)
    ) {
        return KMX_BAD;
    }
    kmx_command(command, 0, KM_HANDLER_COPY, first, count);
    command->a0 = copy.from.address;
    command->a1 = copy.from.bank;
    command->a2 = copy.from.first;
    command->a3 = copy.to.address;
    command->a4 = copy.to.bank;
    command->a5 = copy.to.first;
    command->a6 = copy.page;
    command->a7 = copy.pages;
    command->a8 = copy.buffer;
    command->a9 = copy.chunk;
    return KMX_OK;
}

/* Core 0: the copy's command, the range's lines dropped (or evicted) first unless
 * `clean`, a store's written back; then submitted and waited for or not. */
static int copy_run(
    int id,
    uint64_t offset,
    const kmx_dram* other,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    int fetch,
    int clean,
    uint64_t deadline,
    kmx_event* event,
    uint32_t* handle
) {
    km_command_t command;
    int status = ring_core();
    if (status == KMX_OK) {
        status = copy_command(&command, id, offset, other, bytes, first, count, fetch);
    }
    if (status != KMX_OK) {
        return status;
    }
    if (!clean && g_one_way && bytes >= KMX_EVICT_FROM) {
        kmx_evict();
    } else if (!clean) {
        kmx_flush_at(id, offset, bytes);
    }
    if (handle != 0) {
        return kmx_ring_submit(&g_ring, &command, handle);
    }
    kmx_event own;
    return until_done(&command, deadline, event != 0 ? event : &own);
}

int kmx_fetch(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) {
    return copy_run(id, offset, from, bytes, first, count, 1, 0, deadline, event, 0);
}

int kmx_fetch_clean(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) {
    return copy_run(id, offset, from, bytes, first, count, 1, 1, deadline, event, 0);
}

int kmx_store(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint64_t deadline,
    kmx_event* event
) {
    return copy_run(id, offset, to, bytes, first, count, 0, 0, deadline, event, 0);
}

int kmx_fetch_submit(
    int id,
    uint64_t offset,
    const kmx_dram* from,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
) {
    return (
        handle == 0
            ? KMX_BAD
            : copy_run(id, offset, from, bytes, first, count, 1, 0, 0, 0, handle)
    );
}

int kmx_store_submit(
    int id,
    uint64_t offset,
    const kmx_dram* to,
    uint64_t bytes,
    uint32_t first,
    uint32_t count,
    uint32_t* handle
) {
    return (
        handle == 0
            ? KMX_BAD
            : copy_run(id, offset, to, bytes, first, count, 0, 0, 0, 0, handle)
    );
}

void kmx_result(uint32_t code, const void* words, uint32_t word_count) {
    volatile kmx_result_block* block = result();
    if (word_count > KMX_RESULT_WORDS_MAX) {
        fail(KMX_ERROR_RESULT);
        word_count = KMX_RESULT_WORDS_MAX;
    }
    const uint32_t* source_words = (const uint32_t*)words;
    for (uint32_t word = 0; word < word_count; word++) {
        block->words[word] = source_words[word];
    }
    block->n = word_count;
    block->code = code;
    g_code = code;
    x280_fence_ow();
}

/* A thread may leave out either of these (kmx.h); the start-up calls those it has. */
#pragma weak kmx_main
#pragma weak kmx_program

/* Between commands: the events of commands kmx_program submitted and did not wait for
 * are set aside for kmx_poll / kmx_wait; past KMX_STASH the oldest is dropped. */
static void keep_events(void) {
    kmx_event event;
    while (kmx_ring_take(&g_ring, &event)) {
        if (g_ring.stashed == KMX_STASH) {
            for (uint32_t position = 1; position < KMX_STASH; position++) {
                g_ring.stash[position - 1] = g_ring.stash[position];
            }
            g_ring.stashed--;
        }
        g_ring.stash[g_ring.stashed++] = event;
    }
}

/* Answers the dispatcher's line `number` with `word`, waiting while the ring is full
 * (its events kept meanwhile): KMX_OK, or why it ended (a stop, the deadline). */
static int answer(uint32_t number, uint32_t word) {
    int status = kmx_ring_answer(&g_ring, number, word);
    for (uint32_t tries = 1; status == KMX_FULL; tries++) {
        keep_events(); /* a ring stalled on unread events frees as they are taken */
        const int end = tries % KMX_RING_SPINS == 0 ? must_end(0) : KMX_OK;
        if (end != KMX_OK) {
            return end;
        }
        status = kmx_ring_answer(&g_ring, number, word);
    }
    return status;
}

/* Core 0 with kmx_program, by its deadline: ends every core's deadline (a command has
 * the dispatcher's), answers line 0 (it serves), then each line until a stop. */
static int serve_commands(void) {
    int status = must_end(0);
    if (status != KMX_OK) {
        return status;
    }
    *(volatile uint32_t*)&g_serving = 1;
    x280_fence_ow();
    status = answer(0, 0);
    for (uint32_t spins = 1; status == KMX_OK; spins++) {
        km_command_t command;
        if (kmx_ring_command(&g_ring, &command)) {
            status = answer(command.sequence, kmx_program(&command));
            keep_events();
            spins = 0;
        } else if (spins % KMX_POLL_EVERY == 0) {
            keep_events();
            status = must_end(0);
        }
    }
    return status;
}

/* X280 ranks: windows opened (1, or 2: they failed), line 0 answered, the run taken
 * last, each core's window onto L1 0 (and whether it is this part's), each rank. */
#define KMX_RANKS_TAKE 8u  /* messages a look at the queue takes at most */
#define KMX_RANKS_LOOK 64u /* looks at an empty queue between looks at the run line */
static uint32_t g_ranks_open __attribute__((section(".data"))) = 0;
static uint32_t g_ranks_said __attribute__((section(".data"))) = 0;
static uint32_t g_ranks_last __attribute__((section(".data"))) = 0;
/* the message ranks a forward reaches: the run's, or over islands island 0's part */
static uint32_t g_ranks_reach __attribute__((section(".data"))) = 0;
/* 1 + the core whose inbox place a ROOM give-up left unfilled: its inbox reads in
 * order and stops there, so every later run of the load is refused (BROKEN) */
static uint32_t g_ranks_broken __attribute__((section(".data"))) = 0;
static volatile uint8_t* g_l1[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static uint8_t g_l1_mine[KMX_PORTS_CORES] __attribute__((section(".data"))) = {0};
static volatile uint8_t* g_ranks_status[KM_X280_RANKS_MAX]
    __attribute__((section(".data"))) = {0};
static kmx_messages g_ranks_state[KM_X280_RANKS_MAX]
    __attribute__((section(".data"))) = {{0}};
static void (*g_ranks_function)(int, kmx_messages*, void*)
    __attribute__((section(".data"))) = 0;
static void* g_ranks_arg __attribute__((section(".data"))) = 0;

static inline const volatile kmx_ranks_line* ranks_line(void) {
    return (const volatile kmx_ranks_line*)(lim() + KMX_LIM_RANKS);
}

static inline int ranks_pow2(uint32_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

/* log2 of a power of two, by a loop: the image links no libgcc for __builtin_ctz. */
static uint32_t ranks_shift(uint32_t value) {
    uint32_t shift = 0;
    while (shift < 31u && (value >> shift) != 1u) {
        shift++;
    }
    return shift;
}

/* The lap mark of `position` in a ring of 2^shift (km::messages_mark): 1 to 65535. */
static uint32_t ranks_mark(uint32_t position, uint32_t shift) {
    const uint32_t lap = position >> shift;
    uint32_t folded = (lap & 0xFFFFu) + (lap >> 16);
    folded = folded >= 0xFFFFu ? folded - 0xFFFFu : folded;
    return folded + 1u;
}

/* Whether the ranks line and its area are in range, with a ring to the dispatcher. */
static int ranks_valid(const volatile kmx_ranks_line* line) {
    const uint32_t bytes = line->message_bytes;
    return (
        line->at != 0 && (line->at & 63u) == 0 && line->at >= x280_hart_args_bytes() &&
        (uint64_t)(uintptr_t)lim() + line->at + KMX_RANKS_BYTES <= X280_LIM_END &&
        line->ranks >= 1 && line->ranks <= KM_X280_RANKS_MAX && line->cores >= 1 &&
        line->cores <= KM_X280_RANK_CORE(0) && ranks_pow2(line->cap) &&
        line->cap >= 32u && bytes % 64u == 0 && bytes >= 64u && bytes <= 256u &&
        ranks_pow2(line->inbox_messages) && (line->inbox & 63u) == 0 &&
        line->inbox + line->inbox_messages * bytes <= KMX_L1_BYTES &&
        line->counters >= 1 && line->counters <= KM_COUNTERS_MAX &&
        (line->queue & 63u) == 0 && g_ring.commands != 0
    );
}

/* Core 0, once: a window onto every core's L1 (the port's where it has one) and each
 * rank's non-posted window onto its status line. 1, or 0 for a window that failed. */
static int ranks_open(const volatile kmx_ranks_line* line) {
    const volatile uint8_t* area = lim() + line->at;
    const uint32_t noc = (ports()->flags & KMX_PORTS_NOC1) ? 1u : 0u;
    for (uint32_t core = 0; core < line->cores; core++) {
        if (g_port[core] != 0) {
            g_l1[core] = g_port[core] - ports()->control - KM_CONTROL_X280;
            continue;
        }
        const uint32_t tile = (
            ((const volatile uint32_t*)(area + KMX_RANKS_TILES))[core]
        );
        const uint32_t x = tile & 0xFFFFu, y = tile >> 16;
        if (x > NOC_MAX_COORD || y > NOC_MAX_COORD) {
            return 0;
        }
        noc_tlb_2m_t programmed;
        programmed.data
            [0] = (programmed.data[1] = programmed.data[2] = programmed.data[3] = 0);
        noc_fill_tlb_2m(&programmed, x, y, x, y, 0, NOC_ORDER_POSTED, 0, 0u);
        programmed.noc_sel = noc;
        g_l1[core] = (
            (volatile uint8_t*)noc_configure_tlb_2m_ext(
                KMX_PORT_WINDOW + core, &programmed, 0
            )
        );
        g_l1_mine[core] = 1;
        if (noc_verify_tlb_2m(KMX_PORT_WINDOW + core, &programmed) != 0) {
            return 0;
        }
    }
    for (uint32_t rank = 0; rank < line->ranks; rank++) {
        const uint64_t address = line->lines + 16ull * KM_X280_RANK_CORE(rank);
        noc_tlb_2m_t programmed;
        volatile uint8_t* window = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                KMX_RANKS_WINDOW + rank,
                line->lines_tile & 0xFFFFu,
                line->lines_tile >> 16,
                address & ~(NOC_2M_WINDOW_STRIDE - 1u),
                NOC_ORDER_DEFAULT,
                &programmed
            )
        );
        if (window == 0) {
            return 0;
        }
        g_ranks_status[rank] = window + (address & (NOC_2M_WINDOW_STRIDE - 1u));
    }
    x280_fence_ow();
    return 1;
}

/* Every window ranks_open opened is cleared (after ports_close). */
static void ranks_close(void) {
    x280_fence_ow();
    for (uint32_t core = 0; core < KMX_PORTS_CORES; core++) {
        if (g_l1_mine[core]) {
            noc_clear_tlb_2m(KMX_PORT_WINDOW + core);
            g_l1_mine[core] = 0;
        }
        g_l1[core] = 0;
    }
    for (uint32_t rank = 0; rank < KM_X280_RANKS_MAX; rank++) {
        if (g_ranks_status[rank] != 0) {
            noc_clear_tlb_2m(KMX_RANKS_WINDOW + rank);
            g_ranks_status[rank] = 0;
        }
    }
}

/* One 16-byte store, as port_store, without its fence. */
static void ranks_store(volatile uint8_t* destination, const uint32_t* source) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(source), "r"(destination)
        : "memory", "v8"
    );
}

/* The rank's status line {run, state, third, fourth}: it has landed on return. */
static void ranks_status(
    kmx_messages* messages, uint32_t state, uint32_t third, uint32_t fourth
) {
    const uint32_t words[4] = {messages->run, state, third, fourth};
    ranks_store(g_ranks_status[messages->rank - messages->count], words);
    x280_fence_ow();
}

/* The rank gives up for `where`: its line says so, the dispatcher ends the run. */
static void ranks_give_up(kmx_messages* messages, uint32_t where) {
    if (messages->gave == 0) {
        messages->gave = where;
        ranks_status(messages, KM_MESSAGES_GAVE_UP, messages->head, where);
    }
}

/* The dispatcher's run line, its four words agreeing with a second look at the
 * sequence: 1, or 0 when torn on every try. */
static int ranks_run_line(uint32_t* words) {
    const volatile uint32_t* line = (
        (const volatile uint32_t*)(g_ring.in + KMX_RING_RUN_AFTER)
    );
    for (int attempt = 0; attempt < 4; attempt++) {
        x280_fence_ir();
        for (int word = 0; word < 4; word++) {
            words[word] = line[word];
        }
        x280_fence_ir();
        if (line[KM_X280_RUN_SEQUENCE] == words[KM_X280_RUN_SEQUENCE]) {
            return 1;
        }
    }
    return 0;
}

int kmx_ranks(void) {
    const volatile kmx_ranks_line* line = ranks_line();
    return line->at != 0 ? (int)line->ranks : 0;
}

/* Rank `x280_core`'s part: the program's function, then what it left in its queue. */
static void ranks_part(int x280_core, void* arg) {
    (void)arg;
    kmx_messages* messages = &g_ranks_state[x280_core];
    g_ranks_function(x280_core, messages, g_ranks_arg);
    while (kmx_messages_next(messages) != 0) {
        kmx_messages_consume(messages); /* a message it did not take ends here */
    }
}

int kmx_messages_run(
    void (*function)(int x280_core, kmx_messages* messages, void* arg),
    void* arg,
    uint64_t deadline
) {
    int status = ring_core();
    const volatile kmx_ranks_line* line = ranks_line();
    if (status != KMX_OK || g_in_parallel || !ranks_valid(line)) {
        return status != KMX_OK ? status : KMX_BAD;
    }
    if (g_ranks_open == 0) {
        g_ranks_open = ranks_open(line) ? 1u : 2u;
    }
    if (g_ranks_open != 1) {
        return KMX_BAD;
    }
    if (!g_ranks_said && !g_serving) { /* the dispatcher stores its lines from now on */
        status = answer(0, 0);
        if (status != KMX_OK) {
            return ended(status);
        }
    }
    g_ranks_said = 1;
    uint32_t words[4];
    for (uint32_t spins = 1;; spins++) {
        if (
            ranks_run_line(words) && words[KM_X280_RUN_STATE] == KM_X280_RUN_START &&
            words[KM_X280_RUN_SEQUENCE] != g_ranks_last
        ) {
            break;
        }
        if (spins % KMX_POLL_EVERY == 0) {
            const int end = must_end(0);
            if (end != KMX_OK) {
                return ended(end);
            }
            if (x280_mtime() > deadline) {
                return KMX_LATE;
            }
        }
    }
    g_ranks_last = words[KM_X280_RUN_SEQUENCE];
    const uint32_t group = words[KM_X280_RUN_GROUP], cores = group >> 16;
    const uint32_t spans = words[KM_X280_RUN_HANDLER] >> 8; /* over islands: counts */
    const uint32_t count = (
        spans != 0
            ? spans >> 8
            : cores > line->counters
                ? cores - line->counters
                : 0
    );
    g_ranks_reach = spans != 0 ? spans & 0xFFu : count;
    const uint64_t now = x280_mtime();
    for (uint32_t rank = 0; rank < line->ranks; rank++) {
        kmx_messages* messages = &g_ranks_state[rank];
        messages->run = words[KM_X280_RUN_SEQUENCE];
        messages->rank = count + rank;
        messages->count = count;
        messages->first = group & 0xFFFFu;
        messages->handler = words[KM_X280_RUN_HANDLER] & 0xFFu;
        messages->quota = line->quota;
        messages->sent = messages->consumed = messages->gave = 0;
        messages->left = messages->idle = 0; /* head goes on over a load's runs */
        messages->at = messages->head;
        messages->last = now;
    }
    if (g_ranks_broken != 0) { /* refused at once: the dispatcher ends the run */
        for (uint32_t rank = 0; rank < line->ranks; rank++) {
            ranks_give_up(&g_ranks_state[rank], KMX_GAVE_BROKEN);
        }
        return KMX_BAD;
    }
    g_ranks_function = function;
    g_ranks_arg = arg;
    for (uint32_t rank = 1; rank < line->ranks && status == KMX_OK; rank++) {
        status = kmx_spawn((int)rank, ranks_part, 0);
    }
    if (status == KMX_OK) {
        ranks_part(0, 0);
    }
    for (uint32_t rank = 1; rank < line->ranks; rank++) {
        const int joined = kmx_join((int)rank);
        status = status == KMX_OK ? joined : status;
    }
    for (uint32_t rank = 0; rank < line->ranks && status == KMX_OK; rank++) {
        status = g_ranks_state[rank].gave != 0 ? KMX_BAD : KMX_OK;
    }
    return status;
}

const volatile uint32_t* kmx_messages_next(kmx_messages* messages) {
    const volatile kmx_ranks_line* line = ranks_line();
    const uint32_t bytes = line->message_bytes, mask = line->cap - 1u;
    const uint32_t shift = ranks_shift(line->cap);
    const uint64_t queue = (
        X280_LOCAL_DDR_TILE_BASE + line->queue +
        (uint64_t)(messages->rank - messages->count) * line->cap * bytes
    );
    if (messages->gave != 0) {
        return 0;
    }
    for (uint32_t looks = 1; messages->left == 0; looks++) {
        const uint32_t position = messages->head;
        const volatile uint32_t*
            first = (
                (const volatile uint32_t*)(queue + (uint64_t)(position & mask) * bytes)
            );
        flush((uint64_t)(uintptr_t)first, 64);
        if ((first[0] & 0xFFFFu) == ranks_mark(position, shift)) {
            if (messages
                    ->idle) { /* busy has landed before the first message is taken */
                ranks_status(messages, KM_MESSAGES_BUSY, 0, 0);
                messages->idle = 0;
            }
            flush((uint64_t)(uintptr_t)first + 64u, bytes - 64u);
            uint32_t valid = 1;
            while (valid < KMX_RANKS_TAKE) {
                const uint64_t next = (
                    queue + (uint64_t)((position + valid) & mask) * bytes
                );
                flush(next, bytes);
                if (
                    (*(const volatile uint32_t*)(uintptr_t)next & 0xFFFFu) !=
                    ranks_mark(position + valid, shift)
                ) {
                    break;
                }
                valid++;
            }
            messages->at = position;
            messages->left = valid;
            messages->head = position + valid;
            messages->consumed += valid;
            break;
        }
        const uint64_t now = x280_mtime();
        if (!messages->idle && now - messages->last >= line->idle) {
            ranks_status(
                messages, KM_MESSAGES_IDLE, messages->sent, messages->consumed
            );
            messages->idle = 1;
        }
        if (looks % KMX_RANKS_LOOK == 0) {
            uint32_t words[4];
            if (
                ranks_run_line(words) &&
                (
                    words[KM_X280_RUN_SEQUENCE] != messages->run ||
                    words[KM_X280_RUN_STATE] == KM_X280_RUN_END
                )
            ) {
                return 0; /* the run ended: the dispatcher saw every line quiet */
            }
            if (must_end(core_id()) != KMX_OK) {
                ranks_give_up(messages, KMX_GAVE_STOPPED);
                return 0;
            }
        }
    }
    const volatile uint32_t*
        message = (
            (const volatile uint32_t*)(queue + (uint64_t)(messages->at & mask) * bytes)
        );
    messages->at++;
    messages->left--;
    messages->last = x280_mtime();
    messages->handler = message[0] >> 16 & 0xFFu;
    return message;
}

int kmx_messages_forward(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
) {
    const volatile kmx_ranks_line* line = ranks_line();
    const uint32_t core = messages->first + rank, bytes = line->message_bytes;
    if (messages->gave != 0) {
        return KMX_BAD;
    }
    if (rank >= g_ranks_reach || core >= line->cores || g_l1[core] == 0) {
        ranks_give_up(messages, KMX_GAVE_RANK);
        return KMX_BAD;
    }
    volatile uint8_t* area = lim() + line->at;
    const uint32_t position = (
        x280_amoadd_w((volatile uint32_t*)(area + KMX_RANKS_PLACES) + core, 1u)
    );
    const volatile uint32_t* credit = (
        (const volatile uint32_t*)(
            area + KMX_RANKS_CREDITS + KM_X280_CREDIT_BYTES * core
        )
    );
    if (position - *credit >= line->inbox_messages) { /* the inbox may be full */
        const uint64_t until = x280_mtime() + line->wait;
        while (position - *credit >= line->inbox_messages && x280_mtime() < until) {
            x280_fence_ir();
        }
        if (position - *credit >= line->inbox_messages) {
            g_ranks_broken = core + 1u;
            ranks_give_up(messages, KMX_GAVE_ROOM);
            return KMX_BAD;
        }
    }
    volatile uint8_t* destination = (
        g_l1[core] + line->inbox + (position & (line->inbox_messages - 1u)) * bytes
    );
    for (uint32_t chunk = 1; chunk < bytes / 16u; chunk++) {
        ranks_store(destination + 16u * chunk, message + 4u * chunk);
    }
    x280_fence_ow(); /* the message before its mark */
    const uint32_t head[4] = {
        ranks_mark(position, ranks_shift(line->inbox_messages)) |
            (handler & 0xFFu) << 16,
        message[1],
        message[2],
        message[3],
    };
    ranks_store(destination, head);
    x280_fence_ow();
    messages->sent++;
    messages->last = x280_mtime();
    return KMX_OK;
}

int kmx_messages_send(
    kmx_messages* messages, uint32_t rank, const uint32_t* message, uint32_t handler
) {
    if (messages->quota == 0) {
        return KMX_FULL;
    }
    messages->quota--;
    return kmx_messages_forward(messages, rank, message, handler);
}

void kmx_messages_consume(kmx_messages* messages) {
    messages->quota++;
}

/* Cores 1 to 3: run each spawn of core 0 until kmx_main returns, a stop is asked
 * for, or the run's deadline passes. */
static void serve(uint64_t core) {
    volatile uint64_t* line = spawn_line(core);
    const volatile uint64_t* exit = spawn_line(0);
    uint64_t seen = 0;
    for (uint32_t spins = 0;; spins++) {
        x280_fence_ir();
        if (exit[KMX_SPAWN_EXIT] != 0) {
            return;
        }
        const uint64_t generation = line[KMX_SPAWN_GEN];
        if (generation > seen) {
            x280_fence_ir();
            void (*function)(
                int, void*
            ) = (void (*)(int, void*))(uintptr_t)line[KMX_SPAWN_FUNCTION];
            function((int)core, (void*)(uintptr_t)line[KMX_SPAWN_ARG]);
            x280_fence_iorw();
            line[KMX_SPAWN_DONE] = generation;
            x280_fence_ow();
            seen = generation;
            continue;
        }
        if (spins % KMX_POLL_EVERY == 0 && must_end(core) != KMX_OK) {
            return;
        }
    }
}

/* Opens a window onto the LIM of every other cluster of the group and, on the lead,
 * clears the done lines. 1, or 0 when a window could not be opened. */
static int peers_open(void) {
    const volatile kmx_peers* group = peers();
    const uint32_t rest = others();
    if (group->lead == 0) {
        return 1;
    }
    if (
        group->lead > 3 || !(group->mask >> group->lead & 1u) ||
        (group->mask & ~0xEu) || !(group->mask >> kmx_cluster() & 1u) ||
        (group->all & 63u) || group->all < KMX_LIM_BYTES ||
        (uint64_t)(uintptr_t)lim() + group->all + KMX_ALL_BYTES > X280_LIM_END
    ) {
        return 0;
    }
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if (!(rest >> cluster & 1u)) {
            continue;
        }
        noc_tlb_2m_t programmed;
        volatile uint8_t* mapped = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                KMX_PEER_WINDOW + cluster - 1,
                group->tile[cluster] & 0xFFFFu,
                group->tile[cluster] >> 16,
                X280_LIM_BASE,
                NOC_ORDER_POSTED,
                &programmed
            )
        );
        if (mapped == 0) {
            return 0;
        }
        /* every cluster's block is at the same LIM address */
        g_peer[cluster] = mapped + ((uint64_t)(uintptr_t)lim() - X280_LIM_BASE);
    }
    if (kmx_lead() == kmx_cluster()) {
        for (uint32_t cluster = 1; cluster < 4; cluster++) {
            volatile uint32_t* done_line = all_line(lim(), cluster);
            done_line[KMX_CALL_GEN] = 0;
            done_line[KMX_CALL_NONCE] = 0;
        }
        x280_fence_ow();
    }
    return 1;
}

/* The lead tells the others its kmx_main returned; every window is closed. */
static void peers_close(void) {
    const uint32_t rest = others();
    for (uint32_t cluster = 1; cluster < 4; cluster++) {
        if (g_peer[cluster] != 0 && kmx_lead() == kmx_cluster()) {
            call(cluster, g_all_gen + 1, 0, 0, 1);
        }
        if (rest >> cluster & 1u) {
            noc_clear_tlb_2m(KMX_PEER_WINDOW + cluster - 1);
            g_peer[cluster] = 0;
        }
    }
}

/* Whether the argument block is this run's and every entry is in range. */
static int args_valid(const volatile kmx_args* arguments, uint64_t run) {
    if (
        x280_hart_args_bytes() < KMX_LIM_BYTES || arguments->run_value != run ||
        arguments->magic != KMX_MAGIC || arguments->version != KMX_VERSION ||
        arguments->deadline == 0 || arguments->deadline > KMX_DEADLINE_MAX ||
        arguments->nregions > KMX_REGIONS_MAX || arguments->cluster < 1 ||
        arguments->cluster > 3 || arguments->nwords > KMX_WORDS_MAX ||
        (arguments->check & 63u)
    ) {
        return 0;
    }
    for (
        uint32_t region_index = 0; region_index < arguments->nregions; region_index++
    ) {
        const uint64_t address = arguments->region[region_index].offset;
        const uint64_t bytes = arguments->region[region_index].bytes;
        if (
            (address & 63u) || bytes == 0 || address + bytes < address ||
            address + bytes > (1ull << 32) ||
            (address <= X280_IPI_ACK_OFFSET && X280_IPI_ACK_OFFSET < address + bytes)
        ) {
            return 0;
        }
    }
    return 1;
}

/* Opens the ring the line names: as it is, or for KMX_RING_REMOTE through a window
 * onto the commands and the read line. 1, or 0 when the line is out of range. */
static int ring_open(const volatile kmx_ring_descriptor* descriptor) {
    kmx_ring_descriptor local;
    local.cmd = descriptor->cmd;
    local.consumed = descriptor->consumed;
    local.events = descriptor->events;
    local.read = descriptor->read;
    local.command_depth = descriptor->command_depth;
    local.event_depth = descriptor->event_depth;
    local.flags = descriptor->flags;
    if (local.flags & KMX_RING_REMOTE) {
        const uint64_t base = local.cmd & ~(NOC_2M_WINDOW_STRIDE - 1u);
        const uint32_t tile = descriptor->reserved[0];
        const uint32_t x = tile & 0xFFFFu;
        const uint32_t y = tile >> 16;
        if (
            (local.read & ~(NOC_2M_WINDOW_STRIDE - 1u)) != base || (local.flags & 1u) ||
            local.read + 64u > base + NOC_2M_WINDOW_STRIDE
        ) {
            return 0;
        }
        noc_tlb_2m_t programmed;
        volatile uint8_t* window = (
            (volatile uint8_t*)noc_configure_tlb_2m_checked(
                KMX_RING_WINDOW, x, y, base, NOC_ORDER_POSTED, &programmed
            )
        );
        if (window == 0) {
            return 0;
        }
        if (noc_verify_tlb_2m(KMX_RING_WINDOW, &programmed) != 0) {
            noc_clear_tlb_2m(KMX_RING_WINDOW);
            return 0;
        }
        local.cmd = (uint64_t)(uintptr_t)window + (local.cmd - base);
        local.read = (uint64_t)(uintptr_t)window + (local.read - base);
    }
    if (!kmx_ring_open(&g_ring, &local)) {
        if (local.flags & KMX_RING_REMOTE) {
            noc_clear_tlb_2m(KMX_RING_WINDOW);
        }
        return 0;
    }
    return 1;
}

/* Whether the cluster's cache is one way that core 0 evicts from: WayEnable 0 and bit 0
 * of core 0's way mask (reads only; all ones is a tile that does not answer). */
static uint32_t one_way(void) {
    const uint32_t ways = *(volatile const uint32_t*)(uintptr_t)KMX_CCACHE_WAYENABLE;
    const uint64_t mask = (
        *(volatile const uint64_t*)(uintptr_t)(
            KMX_CCACHE_WAYMASK + 8ull * KMX_CCACHE_MASTER_CORE0
        )
    );
    return ways != 0xFFFFFFFFu && (ways & 0xFu) == 0 && mask != ~0ull && (mask & 1u);
}

/* Core 0: the bank check, kmx_main, the result block. */
static uint64_t run_main(const volatile kmx_args* arguments) {
    volatile kmx_result_block* block = result();
    block->state = KMX_STATE_BOOT;
    x280_fence_ow();
    const volatile kmx_ring_descriptor*
        descriptor = ((const volatile kmx_ring_descriptor*)(lim() + KMX_LIM_RING));
    if (descriptor->cmd != 0 && !ring_open(descriptor)) {
        fail(KMX_ERROR_ARGS); /* a ring is named but its line is out of range */
    }
    if (!peers_open()) {
        fail(KMX_ERROR_ARGS); /* the peers line is out of range */
    }
    if (!ports_open()) {
        fail(KMX_ERROR_ARGS); /* the ports line is out of range, or a window failed */
    }
    g_one_way = one_way();
    const uint64_t check = X280_LOCAL_DDR_TILE_BASE + arguments->check;
    flush(check, 64);
    if (*(volatile uint64_t*)(uintptr_t)check != arguments->check_value) {
        fail(KMX_ERROR_BANK);
    }
    if (block->error == KMX_ERROR_NONE) {
        block->state = KMX_STATE_RUNNING;
        x280_fence_ow();
        const int code = kmx_main != 0 ? kmx_main(0) : 0;
        if (g_code == KMX_UNSET) {
            block->code = (uint32_t)code;
        }
        if (kmx_program != 0) {
            /* a stop is how a thread that serves commands ends */
            const int end = g_ring.commands != 0 ? serve_commands() : KMX_BAD;
            if (end == KMX_LATE || end == KMX_BAD) {
                fail(end == KMX_LATE ? KMX_ERROR_LATE : KMX_ERROR_ARGS);
            }
        }
    }
    if (g_ring.commands != 0 && (descriptor->flags & KMX_RING_REMOTE)) {
        noc_clear_tlb_2m(KMX_RING_WINDOW);
    }
    peers_close();
    ports_close();
    ranks_close();
    const uint64_t ticks = x280_mtime() - g_start;
    block->ticks = (uint32_t)ticks;
    /* every region flushed: a dirty line left in the cluster's cache would be
     * written over a later load's data (or its check line) when it is evicted */
    for (
        uint32_t region_index = 0; region_index < arguments->nregions; region_index++
    ) {
        uint64_t bytes;
        const void* region = kmx_region((int)region_index, &bytes);
        flush(
            (uint64_t)(uintptr_t)region, bytes
        ); /* core 0 alone: as fast, and after a stop */
    }
    spawn_line(0)[KMX_SPAWN_EXIT] = 1;
    x280_fence_ow();
    block->state = block->error ? KMX_STATE_FAILED : KMX_STATE_DONE;
    x280_fence_ow();
    return block->error ? ~(uint64_t)block->error : ticks;
}

static __attribute__((noinline)) void boot(uint64_t core) {
    const uint64_t started = x280_mtime();
    const volatile kmx_args* arguments = (const volatile kmx_args*)lim();
    if (arguments == 0) {
        return;
    }
    const uint64_t run = x280_hart_control(core)->run_value;
    if (!args_valid(arguments, run)) {
        if (core == 0) {
            result()->error = KMX_ERROR_ARGS;
            result()->state = KMX_STATE_FAILED;
            x280_fence_ow();
        }
        x280_hart_set_result(core, ~(uint64_t)KMX_ERROR_ARGS);
        return;
    }
    g_deadline[core] = started + arguments->deadline;
    if (core != 0) {
        serve(core);
        return;
    }
    g_start = started;
    x280_hart_set_result(0, run_main(arguments));
}

int main(uint64_t core_id) {
    /* Floating point and vector on first, on every core: the harness leaves them off
     * and the first such instruction would trap. */
    __asm__ volatile("csrs mstatus, %0" ::"r"((3ul << 9) | (3ul << 13)));
    const volatile kmx_args* arguments = (const volatile kmx_args*)lim();
    if (arguments != 0 && !(arguments->flags & KMX_FLAG_KEEP_PREFETCH)) {
        volatile uint32_t*
            prefetcher = (
                (volatile uint32_t*)(uintptr_t)(
                    KMX_L2PF_BASE + (core_id & 3u) * KMX_L2PF_STRIDE
                )
            );
        prefetcher[0] = KMX_L2PF_BASIC;
        prefetcher[1] = KMX_L2PF_USER;
        x280_fence_iorw();
    }
    boot(core_id);
    return 0;
}
