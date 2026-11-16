// Program multiply, its writer: APPLY answers a0 * a1.
#include "komorebi/device/core/km_program.hpp"
#include "../handlers.h"

void km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) == kHandlerApply) {
        writer.result(command.a0 * command.a1);
    }
}
