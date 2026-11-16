// The runtime's reader of a run of programs: for each command, the part of the
// command's program (loaded when the tile does not hold it), then its call.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/overlays/km_overlay.hpp"
#include "komorebi/device/km_worker.hpp"
#if defined(KM_PROGRAM_CHAIN)
#include "komorebi/device/movement/km_chain.hpp"
#endif

void kernel_main() {
    km::Reader reader;
#if defined(KM_PROGRAM_CHAIN)
    km::chain_start();  // a program here runs chains: its words zero before any command
#endif
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == KM_HANDLER_LOAD) {
            if (!km::program_switch(reader, command)) {
                return;  // the give-up path
            }
        } else if (km::handler(command) - 1u >= KM_HANDLER_USER_MAX) {
            return;  // NOP: no program's
        }
        km::Overlay program(reader, km::program_overlay(km::program(command)));
        if (!program.load()) {
            return;  // the give-up path
        }
        program.call(command);
    });
}
