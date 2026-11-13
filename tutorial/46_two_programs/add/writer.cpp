// Program add, its writer: APPLY answers a0 + a1.
#include "komorebi/device/km_program.hpp"
#include "../handlers.h"

// --8<-- [start:program]
void km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) == kHandlerApply) {
        writer.result(command.a0 + command.a1);
    }
}

// --8<-- [end:program]
