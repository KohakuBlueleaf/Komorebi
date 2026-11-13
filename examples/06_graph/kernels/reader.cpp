// Graph reader: STALL makes rank a0 wait for the stop flag; other handlers need no
// input through the reader.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "graph.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) == kHandlerStall && rd.me() - km::first(r) == r.a0) {
            rd.wait_stop(r);
        }
    });
}
