// Collectives across islands, writer: no handler writes (the reader's calls do).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "collectives.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t&) {
        return true;
    });
}
