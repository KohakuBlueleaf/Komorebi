// The collectives' account, reader: GATHER and REDUCE each the one call.
#include <stdint.h>

#include "account.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"

void kernel_main() {
    km::Reader reader;
    const uint32_t fold = get_write_ptr(kCbFold);
    km::coll_stamps_in(get_write_ptr(kCbStamps));  // with KM_COLL_STAMPS
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerGather) {
            km::all_gather(reader, command, command.a0, command.a1, command.a2);
        } else if (km::handler(command) == kHandlerReduce) {
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
        }
    });
}
