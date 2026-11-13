// Generic reader of a dynamic program: the dynamic handler takes this core's queue;
// every other command (the run's end command among them) is a NOP here.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    const uint32_t dyn_handler = rd.arg(0);
    const bool filled = km::dyn_zero_fill(rd);  // before READY: nobody sends yet
    rd.run([&](const km_command_t& r) {
        if (km::handler(r) == dyn_handler) {
            km::dyn_reader(rd, r, filled);
        }
    });
}
