// The reader kernel: copies one tile from DRAM into the buffer in L1.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t source_address = command.a0;
            const uint32_t source_tile = command.a1;
            const auto source = TensorAccessor(source_arguments, source_address);
            if (!reader.reserve(kTileBuffer, 1)) {
                return;
            }
            noc_async_read_page(source_tile, source, get_write_ptr(kTileBuffer));
            noc_async_read_barrier();
            cb_push_back(kTileBuffer, 1);
        }
    });
}
