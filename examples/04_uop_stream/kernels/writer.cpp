// Writer: MAP stores the core's tile at output page a3 + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "uop_stream.h"

void kernel_main() {
    constexpr auto output_args = TensorAccessorArgs<0>();
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) != kHandlerMap) {
            return true;
        }
        const uint32_t output_address = command.a2;
        const uint32_t first_page = command.a3;
        if (!writer.wait_front(kCbOut, 1)) {
            return false;  // nothing to report: the tile is stopping
        }
        const auto output = TensorAccessor(output_args, output_address);
        noc_async_write_page(
            first_page + writer.me() - km::first(command), output, get_read_ptr(kCbOut)
        );
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
