// Ceilings writer (NoC1): READ's odd chunks of this rank's pages, nothing kept.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ceilings.h"
#include "chunks.hpp"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerRead) {
            read_chunks(writer, command, args, 1, get_write_ptr(kCbWrite));
        }
        return true;
    });
}
