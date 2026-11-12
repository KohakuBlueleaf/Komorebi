// Generic reader of a dynamic program: the dynamic segment takes this core's queue;
// every other record (the run's end record among them) is a NOP here.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    const uint32_t dyn_seg = rd.arg(0);
    const bool filled = km::dyn_zero_fill(rd);  // before READY: nobody sends yet
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) == dyn_seg) {
            km::dyn_reader(rd, r, filled);
        }
    });
}
