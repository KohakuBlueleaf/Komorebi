// The reader kernel: copies two tiles of the source tensor into two buffers.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    constexpr auto source_arguments = TensorAccessorArgs<0>();
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerAdd) {
            const uint32_t source_address = command.a0;
            const uint32_t first_tile = command.a1;
            const uint32_t second_tile = command.a2;
            const auto source = TensorAccessor(source_arguments, source_address);
            if (!reader.reserve(kFirstInput, 1) || !reader.reserve(kSecondInput, 1)) {
                return;
            }
            noc_async_read_page(first_tile, source, get_write_ptr(kFirstInput));
            noc_async_read_page(second_tile, source, get_write_ptr(kSecondInput));
            noc_async_read_barrier();
            cb_push_back(kFirstInput, 1);
            cb_push_back(kSecondInput, 1);
        }
    });
    // --8<-- [end:run]
}
