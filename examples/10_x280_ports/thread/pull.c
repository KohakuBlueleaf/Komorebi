/* X280 ports example, the pull thread: five passes (equal, pulled, logged, replayed,
 * picked), each followed by two reports (README.md). */
#include <stdint.h>

#include "x280_ports.h"
#include "kmx.h"

/* words or the regions out of range, or the log ran out of room */
#define CODE_BAD 1u
#define CODE_LATE 2u /* a run was late or stopped, or the ring stayed full */
#define FULL_TICKS KMX_TICKS_PER_SECOND /* how long a full ring is retried */

static uint32_t* lines(void) {
    uint64_t bytes;
    return (uint32_t*)kmx_region(0, &bytes);
}

/* Fills c field by field (a struct assignment may call memset, which does not link). */
static void command(km_command_t* c, uint32_t handler, uint32_t a0, uint32_t a1) {
    const uint32_t* w = kmx_args_get()->words;
    c->seq = c->land1 = c->land2 = c->land3 = 0;
    c->hdr = handler;
    c->group = 0 | (uint32_t)kmx_port_cores() << 16;
    c->a0 = a0;
    c->a1 = a1;
    c->a2 = w[HP_W_BASE];
    c->a3 = w[HP_W_EXTRA];
    c->a4 = w[HP_W_CENTRE];
    c->a5 = w[HP_W_HALF];
    c->a6 = c->a7 = c->a8 = c->a9 = 0;
}

/* X280 core 0 of every cluster: every line zeroed (it flushes them at the end). */
static void clear(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    volatile uint32_t* out = lines(); /* volatile: no memset, which does not link */
    for (uint32_t i = 0; cpu == 0 && i < HP_LINE * HP_KINDS * KMX_PORTS_CORES; i++) {
        out[i] = 0;
    }
}

/* A pass on ranges dealt by the Tensix cores themselves: nothing to serve. */
static void idle(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)cpu;
    (void)arg;
    while (kmx_port_open()) {
        ;
    }
}

/* One X280 core, a report: each of its cores' tell into the core's line of kind
 * arg >> 1, words from arg & 1 (0: units, work, span; 1: hash, pulls, waited). */
static void collect(int cluster, int cpu, uint64_t arg) {
    uint8_t mine[KMX_PORTS_CORES];
    int n = 0;
    for (int c = 0; c < kmx_port_cores(); c++) {
        if (kmx_port_owner(c) == cpu) {
            mine[n++] = (uint8_t)c;
        }
    }
    uint32_t* out = lines() + HP_LINE * KMX_PORTS_CORES * (arg >> 1);
    const uint32_t at = arg & 1u ? HP_L_HASH : HP_L_UNITS;
    int left = n;
    while (left > 0 && kmx_port_open()) {
        for (int i = 0; i < n; i++) {
            kmx_msg m;
            if (mine[i] != 0xFFu && kmx_port_told(mine[i], &m) == KMX_OK) {
                uint32_t* l = out + HP_LINE * mine[i];
                l[at] = m.w[0];
                l[at + 1] = m.w[1];
                l[at + 2] = m.w[2];
                l[HP_L_WHO] = 1u + ((uint32_t)cluster << 2 | (uint32_t)cpu);
                mine[i] = 0xFFu;
                left--;
            }
        }
    }
}

/* Core c's k-th range in the picked pass: ranges c, c + cores, c + 2 * cores.. */
static uint32_t pick(int core, uint32_t k, uint64_t ranges) {
    const uint64_t j = (uint64_t)core + (uint64_t)k * (uint32_t)kmx_port_cores();
    return j < ranges ? (uint32_t)j : KMX_PULL_NONE;
}

/* X280 core 0 of every cluster: the lines and the log into the bank, for the host. */
static void flush_out(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (cpu == 0) {
        kmx_flush(0);
        kmx_flush(1);
    }
}

/* Runs c (served by fn, or pulled when p is set) until its event, retrying a full ring;
 * its ticks into *dt. A KMX_ code, or KMX_LATE when the command did not complete. */
static int run(
    const km_command_t* c,
    const kmx_pull* p,
    void (*fn)(int, int, uint64_t),
    uint64_t arg,
    uint32_t* dt
) {
    const uint64_t t0 = kmx_time(), until = t0 + FULL_TICKS;
    kmx_event ev;
    int e;
    do {
        e = p ? kmx_pull_run(c, p, &ev) : kmx_port_run(c, fn, arg, &ev);
    } while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested());
    *dt = (uint32_t)(kmx_time() - t0);
    return e == KMX_OK && ev.kind != KM_EV_DONE ? KMX_LATE : e;
}

int kmx_main(int cpu) {
    (void)cpu;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
    const kmx_args* a = kmx_args_get();
    const uint32_t* w = a->words;
    uint64_t out_bytes = 0, log_bytes = 0;
    if (
        a->nwords < HP_WORDS || kmx_region(0, &out_bytes) == 0 ||
        out_bytes < 4u * HP_LINE * HP_KINDS * KMX_PORTS_CORES ||
        kmx_region(1, &log_bytes) == 0 || kmx_port_cores() == 0 || w[HP_W_PER] == 0 ||
        w[HP_W_HALF] == 0
    ) {
        return CODE_BAD;
    }
    const uint64_t ranges = ((uint64_t)w[HP_W_UNITS] + w[HP_W_PER] - 1) / w[HP_W_PER];
    uint32_t ticks[HP_KINDS] = {0, 0, 0, 0, 0};
    if (kmx_par_all(clear, 0) != KMX_OK) {
        return CODE_LATE;
    }
    for (uint32_t k = 0; k < HP_KINDS; k++) {
        static const uint32_t modes[HP_KINDS] = {
            0, KMX_PULL_COUNT, KMX_PULL_LOG, KMX_PULL_REPLAY, KMX_PULL_COUNT
        };
        const kmx_pull p = {
            w[HP_W_UNITS], w[HP_W_PER], modes[k], 1u, k == HP_PICK ? pick : 0, ranges
        };
        km_command_t c;
        command(&c, HX_HANDLER_PASS, w[HP_W_UNITS], k != HP_EQUAL);
        int e = run(&c, k == HP_EQUAL ? 0 : &p, idle, 0, &ticks[k]);
        for (uint32_t page = 0; page < 2 && e == KMX_OK; page++) {
            uint32_t dt;
            command(&c, HX_HANDLER_REPORT, page, 0);
            e = run(&c, 0, collect, k << 1 | page, &dt);
        }
        if (e != KMX_OK) {
            return e == KMX_BAD ? CODE_BAD : CODE_LATE;
        }
    }
    if (kmx_par_all(flush_out, 0) != KMX_OK) {
        return CODE_LATE;
    }
    kmx_result(0, ticks, HP_KINDS);
    return 0;
}
