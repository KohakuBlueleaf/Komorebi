// Commands to an X280 thread, grid side: KM_TC_GRID_RESULT's leader gives tc_grid(a0)
// as its result word; NOP and every other handler do nothing.
#include <stdint.h>

#include "../tc.h"
#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    km::Writer writer;
    writer.run([&](const km_command_t& command) {
        if (
            km::handler(command) == KM_TC_GRID_RESULT &&
            writer.me() == km::first(command)
        ) {
            writer.result(tc_grid(command.a0));
        }
        return true;
    });
}
