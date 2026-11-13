/* A thread that serves commands: no kmx_main, only kmx_program. kHandlerTriangle and
 * kHandlerSquare come from the handlers Python declared for the thread. */
#include "kmx.h"

// --8<-- [start:program]
uint32_t kmx_program(const km_command_t* command) {
    const uint32_t n = command->a0;
    switch (kmx_handler(command)) {
        case kHandlerTriangle:
            return n * (n + 1) / 2;
        case kHandlerSquare:
            return n * n;
    }
    return 0;
}

// --8<-- [end:program]
