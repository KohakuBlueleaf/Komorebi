/* An X280 thread in a run of two programs, add and multiply: BOTH sends APPLY to each
 * program on one core and returns the sum of their answers. */
#include "kmx.h"

/* APPLY of `program` to a0 and a1 on core `core`; its answer, or 0 if it failed. */
// --8<-- [start:apply]
static uint32_t apply(
    uint32_t program, uint32_t handler, uint32_t core, uint32_t a0, uint32_t a1
) {
    km_command_t command;
    kmx_command(&command, program, handler, core, 1);
    command.hdr |= KM_FLAG_RESULT << 8; /* the answer comes back in the event */
    command.a0 = a0;
    command.a1 = a1;
    uint32_t handle = 0;
    kmx_event event;
    if (
        kmx_submit(&command, &handle) != KMX_OK ||
        kmx_wait(handle, kmx_time() + KMX_TICKS_PER_SECOND, &event) != KMX_OK
    ) {
        return 0;
    }
    return event.ticks;
}

// --8<-- [end:apply]

// --8<-- [start:program]
uint32_t kmx_program(const km_command_t* command) {
    if (kmx_handler(command) != kHandlerBoth) {
        return 0;
    }
    const uint32_t sum = (
        apply(kProgramAdd, kProgramAddHandlerApply, 0, command->a0, command->a1)
    );
    const uint32_t product = (
        apply(
            kProgramMultiply, kProgramMultiplyHandlerApply, 1, command->a0, command->a1
        )
    );
    return sum + product;
}

// --8<-- [end:program]
