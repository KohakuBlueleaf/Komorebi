// Work-cluster example, grid side: the resident worker has no segment of its own in
// this first form (the X280 thread does the work), so every record is a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t&) {});
}
