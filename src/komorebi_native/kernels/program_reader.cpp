// The runtime's reader of a run of programs: for each command, the part of the
// command's program (loaded when the tile does not hold it), then its call.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) - 1u >= KM_HANDLER_USER_MAX) {
            return;  // NOP: no program's
        }
        km::Overlay program(reader, km::program_overlay(km::program(command)));
        if (!program.load()) {
            return;  // the give-up path
        }
        program.call(command);
    });
}
