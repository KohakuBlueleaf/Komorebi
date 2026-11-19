/* A thread that serves commands: "hold" answers a0 milliseconds later (past its
 * command's deadline), "sum" at once. */
#include "kmx.h"

uint32_t kmx_program(const km_command_t* command) {
    switch (kmx_handler(command)) {
        case kHandlerHold: {
            const uint64_t until = (
                kmx_time() + (uint64_t)command->a0 * (KMX_TICKS_PER_SECOND / 1000u)
            );
            while (kmx_time() < until && !kmx_stop_requested()) {
                ;
            }
            return 1;
        }
        case kHandlerSum:
            return command->a0 + command->a1;
    }
    return 0;
}
