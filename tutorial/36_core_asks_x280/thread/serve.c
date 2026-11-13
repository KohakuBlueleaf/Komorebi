/* An X280 thread that serves core 0 of the grid: it answers core 0's ask with the
 * square of the number, and returns what core 0 tells it afterwards. */
#include "kmx.h"

__attribute__((section(".data"))) static volatile uint32_t proof = 1;

// --8<-- [start:look]
static void look(void) {
    kmx_message message;
    if (kmx_port_asked(0, &message) == KMX_OK) {
        kmx_port_answer(0, message.sequence, message.w[0] * message.w[0], 0, 0);
    }
    if (kmx_port_told(0, &message) == KMX_OK) {
        proof = message.w[0];
    }
}

// --8<-- [end:look]

// --8<-- [start:serve]
static void serve(int cluster, int cpu, uint64_t argument) {
    (void)cluster;
    (void)argument;
    if (kmx_port_owner(0) != cpu) {
        return;
    }
    while (kmx_port_open()) {
        look();
    }
    look();
}

// --8<-- [end:serve]

int kmx_main(int core) {
    (void)core;
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = 0;
    command.group = 0 | 1 << 16;
    command.a0 = kmx_args_get()->words[0];
    command.a1 = command.a2 = command.a3 = command.a4 = command.a5 = 0;
    command.a6 = command.a7 = command.a8 = command.a9 = 0;

    // --8<-- [start:run]
    kmx_event event;
    if (kmx_port_run(&command, serve, 0, &event) != KMX_OK) {
        return 1;
    }
    // --8<-- [end:run]
    const uint32_t told = proof;
    kmx_result(event.kind == KM_EVENT_DONE ? 0 : 2, &told, 1);
    return 0;
}
