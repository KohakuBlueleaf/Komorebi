// The wide-node probe's writer: nothing to write; the results stay in L1.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "probe.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t&) {
        return true;
    });
}
