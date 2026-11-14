// Ceilings reader (NoC0): READ's even chunks of this rank's pages, nothing kept.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ceilings.h"
#include "chunks.hpp"

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerRead) {
            read_chunks(reader, command, args, 0, get_write_ptr(kCbRead));
        }
    });
}
