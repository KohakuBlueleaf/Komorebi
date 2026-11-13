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

/* Fills the command field by field (a struct assignment may call memset, which does not
 * link). */
static void build_command(
    km_command_t* command,
    uint32_t handler,
    uint32_t first_argument,
    uint32_t second_argument
) {
    const uint32_t* words = kmx_args_get()->words;
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = handler;
    command->group = 0 | (uint32_t)kmx_port_cores() << 16;
    command->a0 = first_argument;
    command->a1 = second_argument;
    command->a2 = words[HP_WORD_BASE];
    command->a3 = words[HP_WORD_EXTRA];
    command->a4 = words[HP_WORD_CENTRE];
    command->a5 = words[HP_WORD_HALF];
    command->a6 = command->a7 = command->a8 = command->a9 = 0;
}

/* X280 core 0 of every cluster: every line zeroed (it flushes them at the end). */
static void clear(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)arg;
    /* volatile: no memset, which does not link */
    volatile uint32_t* line_words = lines();
    for (
        uint32_t i = 0; x280_core == 0 && i < HP_LINE * HP_KINDS * KMX_PORTS_CORES; i++
    ) {
        line_words[i] = 0;
    }
}

/* A pass on ranges dealt by the Tensix cores themselves: nothing to serve. */
static void idle(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)x280_core;
    (void)arg;
    while (kmx_port_open()) {
        ;
    }
}

/* One X280 core, a report: each of its cores' tell into the core's line of kind
 * arg >> 1, words from arg & 1 (0: units, work, span; 1: hash, pulls, waited). */
static void collect(int cluster, int x280_core, uint64_t arg) {
    uint8_t served_cores[KMX_PORTS_CORES];
    int served_count = 0;
    for (int core = 0; core < kmx_port_cores(); core++) {
        if (kmx_port_owner(core) == x280_core) {
            served_cores[served_count++] = (uint8_t)core;
        }
    }
    uint32_t* kind_lines = lines() + HP_LINE * KMX_PORTS_CORES * (arg >> 1);
    const uint32_t first_word = arg & 1u ? HP_LINE_HASH : HP_LINE_UNITS;
    int waiting = served_count;
    while (waiting > 0 && kmx_port_open()) {
        for (int index = 0; index < served_count; index++) {
            kmx_message message;
            if (
                served_cores[index] != 0xFFu &&
                kmx_port_told(served_cores[index], &message) == KMX_OK
            ) {
                uint32_t* line = kind_lines + HP_LINE * served_cores[index];
                line[first_word] = message.w[0];
                line[first_word + 1] = message.w[1];
                line[first_word + 2] = message.w[2];
                line[HP_LINE_SERVED_BY] = (
                    1u + ((uint32_t)cluster << 2 | (uint32_t)x280_core)
                );
                served_cores[index] = 0xFFu;
                waiting--;
            }
        }
    }
}

/* A core's pull_index-th range in the picked pass: ranges core, core + cores, core +
 * 2 * cores, ... */
static uint32_t pick(int core, uint32_t pull_index, uint64_t ranges) {
    const uint64_t range = (
        (uint64_t)core + (uint64_t)pull_index * (uint32_t)kmx_port_cores()
    );
    return range < ranges ? (uint32_t)range : KMX_PULL_NONE;
}

/* X280 core 0 of every cluster: the lines and the log into the bank, for the host. */
static void flush_out(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (x280_core == 0) {
        kmx_flush(0);
        kmx_flush(1);
    }
}

/* Runs the command (served by `serve`, or pulled when `pull` is set) until its event,
 * retrying a full ring; its ticks into *elapsed. A KMX_ code, or KMX_LATE if undone. */
static int run_command(
    const km_command_t* command,
    const kmx_pull* pull,
    void (*serve)(int, int, uint64_t),
    uint64_t arg,
    uint32_t* elapsed
) {
    const uint64_t start = kmx_time(), deadline = start + FULL_TICKS;
    kmx_event event;
    int error;
    do {
        error = (
            pull
                ? kmx_pull_run(command, pull, &event)
                : kmx_port_run(command, serve, arg, &event)
        );
    } while (error == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested());
    *elapsed = (uint32_t)(kmx_time() - start);
    return error == KMX_OK && event.kind != KM_EVENT_DONE ? KMX_LATE : error;
}

int kmx_main(int x280_core) {
    (void)x280_core;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
    const kmx_args* arguments = kmx_args_get();
    const uint32_t* words = arguments->words;
    uint64_t lines_bytes = 0, log_bytes = 0;
    if (
        arguments->nwords < HP_WORDS || kmx_region(0, &lines_bytes) == 0 ||
        lines_bytes < 4u * HP_LINE * HP_KINDS * KMX_PORTS_CORES ||
        kmx_region(1, &log_bytes) == 0 || kmx_port_cores() == 0 ||
        words[HP_WORD_UNITS_PER_RANGE] == 0 || words[HP_WORD_HALF] == 0
    ) {
        return CODE_BAD;
    }
    const uint64_t ranges = (
        ((uint64_t)words[HP_WORD_UNITS] + words[HP_WORD_UNITS_PER_RANGE] - 1) /
        words[HP_WORD_UNITS_PER_RANGE]
    );
    uint32_t ticks[HP_KINDS] = {0, 0, 0, 0, 0};
    if (kmx_parallel_all(clear, 0) != KMX_OK) {
        return CODE_LATE;
    }
    for (uint32_t kind = 0; kind < HP_KINDS; kind++) {
        static const uint32_t modes[HP_KINDS] = {
            0, KMX_PULL_COUNT, KMX_PULL_LOG, KMX_PULL_REPLAY, KMX_PULL_COUNT
        };
        const kmx_pull pull = {
            words[HP_WORD_UNITS],
            words[HP_WORD_UNITS_PER_RANGE],
            modes[kind],
            1u,
            kind == HP_PICK ? pick : 0,
            ranges
        };
        km_command_t command;
        build_command(
            &command, HX_HANDLER_PASS, words[HP_WORD_UNITS], kind != HP_EQUAL
        );
        int error = (
            run_command(&command, kind == HP_EQUAL ? 0 : &pull, idle, 0, &ticks[kind])
        );
        for (uint32_t page = 0; page < 2 && error == KMX_OK; page++) {
            uint32_t report_ticks;
            build_command(&command, HX_HANDLER_REPORT, page, 0);
            error = run_command(&command, 0, collect, kind << 1 | page, &report_ticks);
        }
        if (error != KMX_OK) {
            return error == KMX_BAD ? CODE_BAD : CODE_LATE;
        }
    }
    if (kmx_parallel_all(flush_out, 0) != KMX_OK) {
        return CODE_LATE;
    }
    kmx_result(0, ticks, HP_KINDS);
    return 0;
}
