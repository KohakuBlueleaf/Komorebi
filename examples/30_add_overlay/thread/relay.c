/* Example 30's thread: RELAY sends one command to the grid and returns its event's
 * ticks, so an X280 thread names what was added to the live run as the host does. */
#include <stdint.h>

#include "kmx.h"

#define WAIT_TICKS (KMX_TICKS_PER_SECOND / 2u) /* a command's wait */
#define BAD 0xFFFFFFFFu                        /* the command went wrong */

/* RELAY: a0 the handler, a1 the cores (from 0), a2 to a9 the command's a0 to a7. Its
 * word: the event's ticks, or BAD. */
static uint32_t relay(const km_command_t* command) {
    km_command_t sent;
    kmx_event event;
    kmx_command(&sent, 0, command->a0, 0, command->a1);
    sent.a0 = command->a2;
    sent.a1 = command->a3;
    sent.a2 = command->a4;
    sent.a3 = command->a5;
    sent.a4 = command->a6;
    sent.a5 = command->a7;
    sent.a6 = command->a8;
    sent.a7 = command->a9;
    const uint64_t deadline = kmx_time() + WAIT_TICKS;
    uint32_t handle = 0;
    int status;
    do {
        status = kmx_submit(&sent, &handle);
    } while (status == KMX_FULL && kmx_time() < deadline);
    if (status != KMX_OK || kmx_wait(handle, deadline, &event) != KMX_OK) {
        return BAD;
    }
    return event.kind == KM_EVENT_DONE ? event.ticks : BAD;
}

uint32_t kmx_program(const km_command_t* command) {
    return kmx_handler(command) == kHandlerRelay ? relay(command) : BAD;
}
