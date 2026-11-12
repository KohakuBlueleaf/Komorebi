// Work-cluster example, grid side: every record reports at once.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer wr;
    wr.run([&](const km_record_t&) {
        return true;
    });
}
