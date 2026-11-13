/* The commanded thread: no kmx_main, only kmx_program. The library calls it for each
 * command whose group is this cluster and sends the command's event with its word. */
#include <stdint.h>

#include "kmx.h"
#include "tc.h"

#define WAIT_TICKS (KMX_TICKS_PER_SECOND / 10) /* one grid command, submit and wait */

/* KM_TC_GRID_RESULT of `value` on core `core`, submitted and waited for: its word + 1.
 */
static uint32_t grid_result(uint32_t value, uint32_t core) {
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = KM_TC_GRID_RESULT | KM_FLAG_RESULT << 8;
    command.group = core | 1u << 16;
    command.a0 = value;
    command.a1 = command.a2 = command.a3 = command.a4 = command.a5 = 0;
    command.a6 = command.a7 = command.a8 = command.a9 = 0;
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    uint32_t handle;
    int status = kmx_submit(&command, &handle);
    while (status == KMX_FULL && kmx_time() < deadline && !kmx_stop_requested()) {
        status = kmx_submit(&command, &handle);
    }
    kmx_event event;
    if (status != KMX_OK || kmx_wait(handle, deadline, &event) != KMX_OK) {
        return KM_TC_FAILED;
    }
    return event.kind == KM_EVENT_DONE ? event.ticks + 1u : KM_TC_FAILED;
}

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case KM_TC_ECHO:
            return tc_echo(command->a0, command->a1);
        case KM_TC_SPIN: {
            const uint64_t end = kmx_time() + command->a0;
            while (kmx_time() < end && !kmx_stop_requested()) {
                ;
            }
            return command->a1;
        }
        case KM_TC_SUBMIT:
            return grid_result(command->a0, command->a1);
    }
    return 0;
}
