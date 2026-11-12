// Generic writer of a dynamic program: the dynamic segment sends this core's outgoing
// items; every other record (the run's end record among them) completes as a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_dyn.hpp"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer wr;
    const uint32_t dyn_seg = wr.arg(0);
    wr.run([&](const km_record_t& r) {
        if (km::seg(r) == dyn_seg) {
            return km::dyn_writer(wr, r);
        }
        return true;
    });
}
