// A crossing alone, writer: nothing to do (the reader does it all).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t&) {
        return true;
    });
}
