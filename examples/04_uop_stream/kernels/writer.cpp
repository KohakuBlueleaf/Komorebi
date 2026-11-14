// Writer: MAP stores the core's tile at output page a3 first_page + rank.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_worker.hpp"
#include "uop_stream.h"

void kernel_main() {
    using Output = km::Tensor<0>;
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
        const Output output(output_address);  // its reference word
        if (!km::write_pages(
            writer,
            command,
            output,
            first_page + writer.me() - km::first(command),
            1,
            get_read_ptr(kCbOut)
        )) {
            return false;
        }
        noc_async_writes_flushed();
        cb_pop_front(kCbOut, 1);
        return true;
    });
}
