// Thread cases, grid side: the thread does the work, every command is a NOP.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_command_t&) {});
}
