// The writer kernel: writes the core's sum to the tile of the destination that is its
// rank.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    using Destination = km::Tensor<0>;
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerBroadcast) {
            const uint32_t rank = writer.me() - km::first(command);
            const Destination destination(command.a2);
            if (!writer.wait_front(kOutput, 1)) {
                return false;
            }
            if (!km::write_pages(
                writer, command, destination, rank, 1, get_read_ptr(kOutput)
            )) {
                return false;
            }
            noc_async_write_barrier();
            cb_pop_front(kOutput, 1);
        }
        return true;
    });
}
