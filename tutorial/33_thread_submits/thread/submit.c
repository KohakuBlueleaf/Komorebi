/* An X280 thread that sends one command to core 0 and returns the answer. */
#include "kmx.h"

int kmx_main(int core) {
    (void)core;
    const kmx_args* arguments = kmx_args_get();

    // --8<-- [start:command]
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = KM_FLAG_RESULT << 8;
    command.group = 0 | 1 << 16;
    command.a0 = arguments->words[0];
    command.a1 = arguments->words[1];
    command.a2 = command.a3 = command.a4 = command.a5 = 0;
    command.a6 = command.a7 = command.a8 = command.a9 = 0;
    // --8<-- [end:command]

    // --8<-- [start:send]
    uint32_t handle;
    if (kmx_submit(&command, &handle) != KMX_OK) {
        return 1;
    }
    kmx_event event;
    if (kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) != KMX_OK) {
        return 2;
    }
    // --8<-- [end:send]

    // --8<-- [start:report]
    const uint32_t answer = event.ticks;
    kmx_result(0, &answer, 1);
    return 0;
    // --8<-- [end:report]
}
