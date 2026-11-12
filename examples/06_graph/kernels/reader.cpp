// Graph reader: STALL makes member a0 wait for the stop word; other segments need no
// input through the reader.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "graph.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        if (km::seg(r) == kSegStall && rd.me() - km::first(r) == r.a0) {
            rd.wait_stop(r);
        }
    });
}
