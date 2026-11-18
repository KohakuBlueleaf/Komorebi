// Pages across islands, writer: WHERE answers which island holds a page of a tensor
// named by its reference word.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_placement.hpp"
#include "komorebi/device/km_worker.hpp"
#include "pages.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerWhere) {
            const uint32_t tensor = command.a0;
            const uint32_t page = command.a1;
            writer.result(km::island_of(tensor, page));
        }
        return true;
    });
}
