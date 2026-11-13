// Generic writer of a dynamic program: the dynamic handler sends this core's outgoing
// items; every other command (the run's end command among them) completes as a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer wr;
    const uint32_t dyn_handler = wr.arg(0);
    wr.run([&](const km_command_t& r) {
        if (km::handler(r) == dyn_handler) {
            return km::dyn_writer(wr, r);
        }
        return true;
    });
}
