// The writer of the load: RUN has no writer work. A handler added later has its own
// writer part, which the runtime loads and calls before this body would run.
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "add.h"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t&) {
        return true;
    });
}
