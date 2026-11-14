/* An X280 thread that hands out the numbers 0 to 63 in ranges of 4 to four cores, and
 * with each range a word: 1 for even ranges, 2 for odd ones. */
#include "kmx.h"

#define CORES 4u
#define NUMBERS 64u
#define PER 4u

// --8<-- [start:pick]
static uint64_t pick_word(int core, uint32_t index, uint64_t ranges) {
    const uint32_t range = (uint32_t)core + index * CORES;
    if (range >= ranges) {
        return KMX_PULL_WORD(KMX_PULL_NONE, 0);
    }
    return KMX_PULL_WORD(range, range % 2 == 0 ? 1 : 2);
}

// --8<-- [end:pick]

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = 0;
    command.group = 0 | CORES << 16;
    command.a0 = command.a1 = command.a2 = command.a3 = command.a4 = command.a5 = 0;
    command.a6 = command.a7 = command.a8 = command.a9 = 0;

    // --8<-- [start:run]
    const kmx_pull pull = {
        NUMBERS, PER, KMX_PULL_COUNT, 0, 0, NUMBERS / PER, pick_word
    };
    kmx_event event;
    if (kmx_pull_run(&command, &pull, &event) != KMX_OK) {
        return 1;
    }
    // --8<-- [end:run]
    uint32_t totals[CORES] = {0, 0, 0, 0};
    for (uint32_t served = 0; served < CORES; served++) {
        kmx_message message;
        if (kmx_port_told((int)served, &message) == KMX_OK) {
            totals[served] = message.w[0];
        }
    }
    kmx_result(event.kind == KM_EVENT_DONE ? 0 : 2, totals, CORES);
    return 0;
}
