// Graph reader: STALL makes rank a0 stalled_rank wait for the stop flag; other handlers
// need no input through the reader.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "graph.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t stalled_rank = command.a0;
        if (
            km::handler(command) == kHandlerStall &&
            reader.me() - km::first(command) == stalled_rank
        ) {
            reader.wait_stop(command);
        }
    });
}
