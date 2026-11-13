// Work-cluster example, grid side: no handler needs input through the reader.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_command_t&) {});
}
