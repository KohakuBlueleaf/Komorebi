// The reader kernel: the sender of the row reads each block from DRAM and multicasts
// it to the row; every other core waits for each block.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_broadcast.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerBroadcast) {
            return;
        }
        km::Broadcast row(reader, command, 0);
        if (!row.ok() || !row.included()) {
            return;
        }
        const auto source = TensorAccessor(source_arguments, command.a0);
        const uint32_t block_count = command.a1;
        for (uint32_t block = 0; block < block_count; block++) {
            if (!row.sender()) {
                if (!row.arrive(block)) {
                    return;
                }
                continue;
            }
            const uint32_t address = row.reserve(block);
            if (address == 0) {
                return;
            }
            noc_async_read_page(block, source, address);
            if (!row.cast(block)) {
                return;
            }
        }
        row.finish();
    });
    // --8<-- [end:run]
}
