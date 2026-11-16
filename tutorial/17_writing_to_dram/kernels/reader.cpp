// The reader kernel: copies one tile from DRAM into the buffer in L1.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerCopy) {
            const uint32_t source_address = command.a0;
            const uint32_t source_tile = command.a1;
            const Source source(source_address);
            if (!reader.reserve(kTileBuffer, 1)) {
                return;
            }
            if (!km::read_pages(
                reader, command, source, source_tile, 1, get_write_ptr(kTileBuffer)
            )) {
                return;
            }
            noc_async_read_barrier();
            cb_push_back(kTileBuffer, 1);
        }
    });
}
