/* The thread's apply command: APPLY (kernels/apply.h) on the cores from the update
 * region's header (frame.h RF_U_*); its ticks, count and code left in the header. */
#include <stdint.h>

#include "frame.h"
#include "kmx.h"

#define APPLY_WAIT_TICKS KMX_TICKS_PER_SECOND /* longer than the command's deadline */

uint32_t serve_apply(void) {
    uint64_t bytes;
    volatile uint32_t* words = (volatile uint32_t*)kmx_region(RF_R_UPDATE, &bytes);
    if (words == 0 || bytes < RF_UPDATE_BYTES) {
        return RF_BAD_ARGS;
    }
    kmx_invalidate_at(RF_R_UPDATE, 0, 4u * RF_U_OTHER);
    const uint32_t group = words[RF_U_GROUP];
    if (group == 0) {
        return RF_NO_APPLY;
    }
    km_command_t command; /* field by field: a struct assignment may call memset */
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = RF_HANDLER_APPLY;
    command.group = group;
    command.a0 = words[RF_U_A0 + 0];
    command.a1 = words[RF_U_A0 + 1];
    command.a2 = words[RF_U_A0 + 2];
    command.a3 = words[RF_U_A0 + 3];
    command.a4 = words[RF_U_A0 + 4];
    command.a5 = words[RF_U_A0 + 5];
    command.a6 = words[RF_U_A0 + 6];
    command.a7 = words[RF_U_A0 + 7];
    command.a8 = words[RF_U_A0 + 8];
    command.a9 = words[RF_U_A0 + 9];
    const uint64_t started = kmx_time();
    const uint64_t until = started + APPLY_WAIT_TICKS;
    uint32_t handle = 0, code = 0;
    int error = kmx_submit(&command, &handle);
    while (error == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        error = kmx_submit(&command, &handle);
    }
    if (error != KMX_OK) {
        code = RF_RING;
    } else {
        kmx_event event;
        if (kmx_wait(handle, kmx_time() + APPLY_WAIT_TICKS, &event) != KMX_OK) {
            code = RF_RING;
        } else if (event.kind != KM_EVENT_DONE) {
            code = RF_COMMAND;
        }
    }
    const uint32_t ticks = (uint32_t)(kmx_time() - started);
    const uint32_t least = words[RF_U_TICKS_LEAST];
    words[RF_U_TICKS] = ticks;
    words[RF_U_TICKS_SUM] = words[RF_U_TICKS_SUM] + ticks;
    words[RF_U_TICKS_LEAST] = words[RF_U_SERVED] == 0 || ticks < least ? ticks : least;
    words[RF_U_SERVED] = words[RF_U_SERVED] + 1;
    words[RF_U_CODE] = code;
    kmx_flush_at(RF_R_UPDATE, 0, 4u * RF_U_OTHER);
    return code;
}
