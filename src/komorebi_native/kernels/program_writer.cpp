// The runtime's writer of a run of programs: the command's program's part meets the
// reader's load and is called; it completes the command unless the part said not to.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == KM_HANDLER_LOAD) {
            // a load of a program: parked until the reader set up, then its own view
            const uint32_t control = writer.args().control;
            if (!km::program_follow(
                control, 0, command.sequence, writer.args().wait_limit
            )) {
                return false;
            }
            setup_local_cb_read_write_interfaces<true, true, false>(
                km::program_buffers(control),
                0,
                km::program_line(control)[km::kSwitchMask]
            );
        } else if (km::handler(command) - 1u >= KM_HANDLER_USER_MAX) {
            return true;  // NOP: no program's
        }
        km::Overlay program(writer, km::program_overlay(km::program(command)));
        if (!program.load()) {
            return false;  // the give-up path
        }
        return program.call(command) != 0;
    });
}
