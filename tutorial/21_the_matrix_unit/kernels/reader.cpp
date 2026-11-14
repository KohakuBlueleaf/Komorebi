// The reader kernel: copies two tiles of the source tensor into two buffers.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Source = km::Tensor<0>;
    km::Reader reader;
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerMultiply) {
            const uint32_t source_address = command.a0;
            const uint32_t first_tile = command.a1;
            const uint32_t second_tile = command.a2;
            const Source source(source_address);
            if (!reader.reserve(kFirstInput, 1) || !reader.reserve(kSecondInput, 1)) {
                return;
            }
            if (!km::read_pages(
                reader, command, source, first_tile, 1, get_write_ptr(kFirstInput)
            )) {
                return;
            }
            if (!km::read_pages(
                reader, command, source, second_tile, 1, get_write_ptr(kSecondInput)
            )) {
                return;
            }
            noc_async_read_barrier();
            cb_push_back(kFirstInput, 1);
            cb_push_back(kSecondInput, 1);
        }
    });
    // --8<-- [end:run]
}
