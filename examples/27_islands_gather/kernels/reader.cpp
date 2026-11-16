// All-gather, reader: every rank of GATHER makes the one call; its island's ranks
// return together once their island's copy holds every island's share.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "gather.h"
#include "komorebi/device/tensors/km_ask.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerGather) {
            return;
        }
        km::all_gather(reader, command, command.a0, command.a1, command.a2);
    });
}
