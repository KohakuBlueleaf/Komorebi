// The writer kernel: writes each tile of this core's share to the destination tensor.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Destination = km::Tensor<0>;
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerMap) {
            const uint32_t destination_address = command.a2;
            const uint32_t first_tile = command.a3;
            const uint32_t rank = writer.me() - km::first(command);
            const km::Partition share = (
                km::partition(command.a1, rank, km::count(command))
            );
            const Destination destination(destination_address);
            for (uint32_t index = 0; index < share.len; index++) {
                if (!writer.wait_front(kOutput, 1)) {
                    return false;
                }
                if (!km::write_pages(
                    writer,
                    command,
                    destination,
                    first_tile + share.start + index,
                    1,
                    get_read_ptr(kOutput)
                )) {
                    return false;
                }
                noc_async_write_barrier();
                cb_pop_front(kOutput, 1);
            }
        }
        return true;
    });
    // --8<-- [end:run]
}
