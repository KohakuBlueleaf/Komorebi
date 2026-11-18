// Far order, writer: no handler of its own.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "order.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t&) {
        return true;
    });
}
