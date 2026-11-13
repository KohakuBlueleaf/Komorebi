// The reader kernel: for each tile of this core's share, copies tile i of the first
// half of the source and tile i of the second half into the two input buffers.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerMap) {
            const uint32_t source_address = command.a0;
            const uint32_t tile_count = command.a1;
            const uint32_t rank = reader.me() - km::first(command);
            const km::Partition share = (
                km::partition(tile_count, rank, km::count(command))
            );
            const auto source = TensorAccessor(source_arguments, source_address);
            for (uint32_t index = 0; index < share.len; index++) {
                const uint32_t tile = share.start + index;
                if (!reader.reserve(kFirstInput, 1) ||
                    !reader.reserve(kSecondInput, 1)) {
                    return;
                }
                noc_async_read_page(tile, source, get_write_ptr(kFirstInput));
                noc_async_read_page(
                    tile_count + tile, source, get_write_ptr(kSecondInput)
                );
                noc_async_read_barrier();
                cb_push_back(kFirstInput, 1);
                cb_push_back(kSecondInput, 1);
            }
        }
    });
    // --8<-- [end:run]
}
