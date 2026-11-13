// The reader kernel: copies tile_count tiles from DRAM through the buffer in L1. When
// the command's a5 is 1 it leaves out cb_push_back, on purpose: a bug to look for.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t source_address = command.a0;
            const uint32_t first_tile = command.a1;
            const uint32_t tile_count = command.a4;
            const bool forget_push = command.a5 == kBugForgetPush;
            const auto source = TensorAccessor(source_arguments, source_address);
            for (uint32_t index = 0; index < tile_count; index++) {
                if (!reader.reserve(kTileBuffer, 1)) {
                    return;
                }
                noc_async_read_page(
                    first_tile + index, source, get_write_ptr(kTileBuffer)
                );
                noc_async_read_barrier();
                if (!forget_push) {
                    cb_push_back(kTileBuffer, 1);
                }
            }
        }
    });
    // --8<-- [end:run]
}
