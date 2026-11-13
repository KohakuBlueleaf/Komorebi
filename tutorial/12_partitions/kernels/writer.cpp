// The writer kernel: works out which items the core of a given rank takes, and sends
// the first item and the number of items back in one word.
#include "komorebi/device/km_worker.hpp"
#include "handlers.h"

void kernel_main() {
    km::Writer writer;
    // --8<-- [start:run]
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerShare) {
            const uint32_t items = command.a0;
            const uint32_t rank = command.a1;
            const km::Partition share = km::partition(items, rank, km::count(command));
            const uint32_t first_item = share.start;
            const uint32_t item_count = share.len;
            writer.result(first_item << 16 | item_count);
        }
        return true;
    });
    // --8<-- [end:run]
}
