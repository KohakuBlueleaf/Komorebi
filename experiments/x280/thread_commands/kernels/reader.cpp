// Commands to an X280 thread, grid side: the reader has nothing to do.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t&) {});
}
