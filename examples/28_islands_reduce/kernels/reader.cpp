// All-reduce, reader: every rank of REDUCE makes the one call; its island's ranks
// return together once their island's copy holds the sum of every island's partial.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "reduce.h"

void kernel_main() {
    km::Reader reader;
    const uint32_t fold = get_write_ptr(kCbFold);  // this core's, never pushed
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerReduce) {
            return;
        }
        km::all_reduce(
            reader,
            command,
            command.a0,
            command.a1,
            command.a2,
            command.a3,
            static_cast<km::Sum>(command.a4),
            fold,
            kFoldPages * kFoldBytes
        );
    });
}
