// Islands experiment, reader: no handler takes input.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t&) {});
}
