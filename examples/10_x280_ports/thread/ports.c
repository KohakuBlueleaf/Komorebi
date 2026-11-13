/* X280 ports example, the thread: per round one command to cores [0, cores), served by
 * every X280 core of every cluster through kmx_port_run until its event (README.md). */
#include <stdint.h>

#include "x280_ports.h"
#include "kmx.h"

#define CODE_BAD 1u  /* words or the region out of range */
#define CODE_LATE 2u /* a run was late or stopped, or the ring stayed full */
#define FULL_TICKS KMX_TICKS_PER_SECOND /* how long a full ring is retried */

/* Result words. */
enum { R_DONE, R_TIMEOUTS, R_TICKS_LO, R_TICKS_HI, R_BEST, R_WORST, R_COUNT };

/* The command's cores: the word, or 0 for every core of the port tables. */
static uint32_t cores(void) {
    const uint32_t n = kmx_args_get()->words[HX_W_CORES];
    return n != 0 ? n : (uint32_t)kmx_port_cores();
}

static uint32_t* lines(void) {
    uint64_t bytes;
    return (uint32_t*)kmx_region(0, &bytes);
}

/* Fills r field by field (a struct assignment may call memset, which does not link). */
static void command(km_command_t* r, uint32_t n, uint32_t asks) {
    r->seq = r->land1 = r->land2 = r->land3 = 0;
    r->hdr = HX_HANDLER_WORK;
    r->group = 0 | n << 16;
    r->a0 = asks;
    r->a1 = r->a2 = r->a3 = r->a4 = r->a5 = r->a6 = r->a7 = r->a8 = r->a9 = 0;
}

/* X280 core 0 of every cluster: every line zeroed (it flushes them at the end). */
static void clear(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    volatile uint32_t* out = lines(); /* volatile: no memset, which does not link */
    for (uint32_t i = 0; cpu == 0 && i < HX_LINE * KMX_PORTS_CORES; i++) {
        out[i] = 0;
    }
}

/* One X280 core in round `round`: its cores' salts, then their asks answered until each
 * told its hash or the command ended; each hash and its times into the core's line. */
static void serve(int cluster, int cpu, uint64_t round) {
    const uint32_t asks = kmx_args_get()->words[HX_W_ASKS];
    uint32_t* out = lines();
    uint8_t mine[KMX_PORTS_CORES];
    uint8_t told[KMX_PORTS_CORES];
    int n = 0;
    for (int c = 0; c < (int)cores(); c++) {
        if (kmx_port_owner(c) == cpu) {
            told[n] = 0;
            mine[n++] = (uint8_t)c;
        }
    }
    for (int i = 0; i < n; i++) {
        kmx_port_tell(mine[i], HX_SALT((uint32_t)round, mine[i]), (uint32_t)round, 0);
    }
    int left = n;
    while (left > 0 && kmx_port_open()) {
        for (int i = 0; i < n; i++) {
            const int c = mine[i];
            kmx_msg m;
            if (kmx_port_asked(c, &m) == KMX_OK) {
                kmx_port_answer(c, m.seq, HX_VALUE(m.w[0], (uint32_t)c), 0, 0);
            }
            if (!told[i] && kmx_port_told(c, &m) == KMX_OK) {
                uint32_t* l = out + HX_LINE * c;
                const uint64_t sum = (
                    ((uint64_t)l[HX_L_SUM_HI] << 32 | l[HX_L_SUM_LO]) + m.w[1]
                );
                l[HX_L_HASH] = m.w[0];
                l[HX_L_ROUNDS] += 1;
                l[HX_L_CALLS] += asks;
                l[HX_L_SUM_LO] = (uint32_t)sum;
                l[HX_L_SUM_HI] = (uint32_t)(sum >> 32);
                l[HX_L_MAX] = m.w[2] > l[HX_L_MAX] ? m.w[2] : l[HX_L_MAX];
                l[HX_L_WHO] = 1u + ((uint32_t)cluster << 2 | (uint32_t)cpu);
                told[i] = 1;
                left--;
            }
        }
    }
}

/* X280 core 0 of every cluster: the lines into the bank, for the host. */
static void flush_lines(int cluster, int cpu, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (cpu == 0) {
        kmx_flush(0);
    }
}

int kmx_main(int cpu) {
    (void)cpu;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
    const kmx_args* a = kmx_args_get();
    uint64_t bytes = 0;
    if (
        a->nwords < HX_WORDS || kmx_region(0, &bytes) == 0 ||
        bytes < 4u * HX_LINE * KMX_PORTS_CORES || kmx_port_cores() == 0 ||
        cores() > (uint32_t)kmx_port_cores()
    ) {
        return CODE_BAD;
    }
    uint32_t w[R_COUNT] = {0, 0, 0, 0, 0xFFFFFFFFu, 0};
    uint64_t ticks = 0;
    if (kmx_par_all(clear, 0) != KMX_OK) {
        return CODE_LATE;
    }
    for (uint32_t k = 1; k <= a->words[HX_W_ROUNDS]; k++) {
        km_command_t r;
        command(&r, cores(), a->words[HX_W_ASKS]);
        kmx_event ev;
        const uint64_t t0 = kmx_time(), until = t0 + FULL_TICKS;
        int e = kmx_port_run(&r, serve, k, &ev);
        while (e == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
            e = kmx_port_run(&r, serve, k, &ev);
        }
        if (e != KMX_OK) {
            return CODE_LATE;
        }
        const uint32_t dt = (uint32_t)(kmx_time() - t0);
        ticks += dt;
        w[R_BEST] = dt < w[R_BEST] ? dt : w[R_BEST];
        w[R_WORST] = dt > w[R_WORST] ? dt : w[R_WORST];
        if (ev.kind != KM_EV_DONE) {
            w[R_TIMEOUTS]++;
            break;
        }
        w[R_DONE]++;
    }
    w[R_TICKS_LO] = (uint32_t)ticks;
    w[R_TICKS_HI] = (uint32_t)(ticks >> 32);
    if (kmx_par_all(flush_lines, 0) != KMX_OK) {
        return CODE_LATE;
    }
    kmx_result(0, w, R_COUNT);
    return 0;
}
