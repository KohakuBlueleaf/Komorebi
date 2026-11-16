// The reader kernel: it asks the X280 for the square of its argument, and tells the
// X280 the answer plus one, as proof that the answer arrived.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/core/km_x280.hpp"

void kernel_main() {
    km::Reader reader;
    // --8<-- [start:port]
    km::X280Port port(reader);
    // --8<-- [end:port]
    // --8<-- [start:run]
    reader.run([&](const km_command_t& command) {
        const uint32_t number = command.a0;
        km::X280Msg answer;
        if (!port.call(number, 0, 0, answer)) {
            return;
        }
        port.tell(answer.w[0] + 1, 0, 0);
    });
    // --8<-- [end:run]
}
