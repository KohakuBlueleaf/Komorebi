/* X280 ports example, the thread: per round one command to cores [0, cores), served by
 * every X280 core of every cluster through kmx_port_run until its event (README.md). */
#include <stdint.h>

#include "x280_ports.h"
#include "kmx.h"

#define CODE_BAD 1u  /* words or the region out of range */
#define CODE_LATE 2u /* a run was late or stopped, or the ring stayed full */
#define FULL_TICKS KMX_TICKS_PER_SECOND /* how long a full ring is retried */

/* Result words. */
enum {
    RESULT_DONE,
    RESULT_TIMEOUTS,
    RESULT_TICKS_LOW,
    RESULT_TICKS_HIGH,
    RESULT_BEST,
    RESULT_WORST,
    RESULT_COUNT
};

/* The command's cores: the word, or 0 for every core of the port tables. */
static uint32_t cores(void) {
    const uint32_t requested = kmx_args_get()->words[HX_WORD_CORES];
    return requested != 0 ? requested : (uint32_t)kmx_port_cores();
}

static uint32_t* lines(void) {
    uint64_t bytes;
    return (uint32_t*)kmx_region(0, &bytes);
}

/* Fills the command field by field (a struct assignment may call memset, which does not
 * link). */
static void build_command(km_command_t* command, uint32_t core_count, uint32_t asks) {
    command->sequence = command->land1 = command->land2 = command->land3 = 0;
    command->hdr = HX_HANDLER_WORK;
    command->group = 0 | core_count << 16;
    command->a0 = asks;
    command->a1 = command->a2 = command->a3 = command->a4 = command->a5 = 0;
    command->a6 = command->a7 = command->a8 = command->a9 = 0;
}

/* X280 core 0 of every cluster: every line zeroed (it flushes them at the end). */
static void clear(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)arg;
    /* volatile: no memset, which does not link */
    volatile uint32_t* line_words = lines();
    for (uint32_t i = 0; x280_core == 0 && i < HX_LINE * KMX_PORTS_CORES; i++) {
        line_words[i] = 0;
    }
}

/* One X280 core in round `round`: its cores' salts, then their asks answered until each
 * told its hash or the command ended; each hash and its times into the core's line. */
static void serve(int cluster, int x280_core, uint64_t round) {
    const uint32_t asks = kmx_args_get()->words[HX_WORD_ASKS];
    uint32_t* core_lines = lines();
    uint8_t served_cores[KMX_PORTS_CORES];
    uint8_t told[KMX_PORTS_CORES];
    int served_count = 0;
    for (int core = 0; core < (int)cores(); core++) {
        if (kmx_port_owner(core) == x280_core) {
            told[served_count] = 0;
            served_cores[served_count++] = (uint8_t)core;
        }
    }
    for (int index = 0; index < served_count; index++) {
        kmx_port_tell(
            served_cores[index],
            HX_SALT((uint32_t)round, served_cores[index]),
            (uint32_t)round,
            0
        );
    }
    int waiting = served_count;
    while (waiting > 0 && kmx_port_open()) {
        for (int index = 0; index < served_count; index++) {
            const int core = served_cores[index];
            kmx_message message;
            if (kmx_port_asked(core, &message) == KMX_OK) {
                kmx_port_answer(
                    core, message.sequence, HX_VALUE(message.w[0], (uint32_t)core), 0, 0
                );
            }
            if (!told[index] && kmx_port_told(core, &message) == KMX_OK) {
                uint32_t* line = core_lines + HX_LINE * core;
                const uint64_t sum = (
                    ((uint64_t)line[HX_LINE_SUM_HIGH] << 32 | line[HX_LINE_SUM_LOW]) +
                    message.w[1]
                );
                line[HX_LINE_HASH] = message.w[0];
                line[HX_LINE_ROUNDS] += 1;
                line[HX_LINE_CALLS] += asks;
                line[HX_LINE_SUM_LOW] = (uint32_t)sum;
                line[HX_LINE_SUM_HIGH] = (uint32_t)(sum >> 32);
                line[HX_LINE_LARGEST] = (
                    message.w[2] > line[HX_LINE_LARGEST]
                        ? message.w[2]
                        : line[HX_LINE_LARGEST]
                );
                line[HX_LINE_SERVED_BY] = (
                    1u + ((uint32_t)cluster << 2 | (uint32_t)x280_core)
                );
                told[index] = 1;
                waiting--;
            }
        }
    }
}

/* X280 core 0 of every cluster: the lines into the bank, for the host. */
static void flush_lines(int cluster, int x280_core, uint64_t arg) {
    (void)cluster;
    (void)arg;
    if (x280_core == 0) {
        kmx_flush(0);
    }
}

int kmx_main(int x280_core) {
    (void)x280_core;
    if (kmx_cluster() != kmx_lead()) {
        return kmx_serve() == KMX_OK ? 0 : (int)CODE_LATE;
    }
    const kmx_args* arguments = kmx_args_get();
    uint64_t bytes = 0;
    if (
        arguments->nwords < HX_WORDS || kmx_region(0, &bytes) == 0 ||
        bytes < 4u * HX_LINE * KMX_PORTS_CORES || kmx_port_cores() == 0 ||
        cores() > (uint32_t)kmx_port_cores()
    ) {
        return CODE_BAD;
    }
    uint32_t result_words[RESULT_COUNT] = {0, 0, 0, 0, 0xFFFFFFFFu, 0};
    uint64_t ticks = 0;
    if (kmx_parallel_all(clear, 0) != KMX_OK) {
        return CODE_LATE;
    }
    for (uint32_t round = 1; round <= arguments->words[HX_WORD_ROUNDS]; round++) {
        km_command_t command;
        build_command(&command, cores(), arguments->words[HX_WORD_ASKS]);
        kmx_event event;
        const uint64_t start = kmx_time(), deadline = start + FULL_TICKS;
        int error = kmx_port_run(&command, serve, round, &event);
        while (error == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
            error = kmx_port_run(&command, serve, round, &event);
        }
        if (error != KMX_OK) {
            return CODE_LATE;
        }
        const uint32_t round_ticks = (uint32_t)(kmx_time() - start);
        ticks += round_ticks;
        result_words[RESULT_BEST] = (
            round_ticks < result_words[RESULT_BEST]
                ? round_ticks
                : result_words[RESULT_BEST]
        );
        result_words[RESULT_WORST] = (
            round_ticks > result_words[RESULT_WORST]
                ? round_ticks
                : result_words[RESULT_WORST]
        );
        if (event.kind != KM_EVENT_DONE) {
            result_words[RESULT_TIMEOUTS]++;
            break;
        }
        result_words[RESULT_DONE]++;
    }
    result_words[RESULT_TICKS_LOW] = (uint32_t)ticks;
    result_words[RESULT_TICKS_HIGH] = (uint32_t)(ticks >> 32);
    if (kmx_parallel_all(flush_lines, 0) != KMX_OK) {
        return CODE_LATE;
    }
    kmx_result(0, result_words, RESULT_COUNT);
    return 0;
}
